from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import SteelConfig
from dataset import SteelDataset, build_pivot, get_transforms, load_train_df
from loss import DiceBCELoss, dice_score
from segmentation import ViTSegmenter


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def train(args: argparse.Namespace) -> None:
    cfg = SteelConfig(
        data_dir=args.data_dir,
        train_csv=os.path.join(args.data_dir, "train.csv"),
        image_dir=os.path.join(args.data_dir, "train_images"),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    df = load_train_df(cfg.train_csv)
    pivot = build_pivot(df)

    n_val = max(1, int(len(pivot) * cfg.val_split))
    n_train = len(pivot) - n_val
    train_pivot, val_pivot = pivot.iloc[:n_train], pivot.iloc[n_train:]

    train_tf = get_transforms(cfg.image_height, cfg.image_width, phase="train")
    val_tf = get_transforms(cfg.image_height, cfg.image_width, phase="val")

    train_ds = SteelDataset(train_pivot, cfg.image_dir, train_tf)
    val_ds = SteelDataset(val_pivot, cfg.image_dir, val_tf)

    train_loader = _make_loader(train_ds, cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader = _make_loader(val_ds, cfg.batch_size, cfg.num_workers, shuffle=False)

    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    vit_cfg = cfg.vit_config()
    model = ViTSegmenter(vit_cfg, num_seg_classes=4).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    # ------------------------------------------------------------------
    # Optimiser, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate / 100
    )
    criterion = DiceBCELoss()
    scaler = GradScaler()

    best_dice = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [train]", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= len(train_loader)

        # ---- validate ----
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [val]", leave=False):
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(images)
                    loss = criterion(logits, masks)
                val_loss += loss.item()
                val_dice += dice_score(logits, masks, threshold=cfg.threshold)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f}"
        )

        ckpt = {"epoch": epoch, "model": model.state_dict(), "cfg": cfg}
        torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "last.pt"))

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "best.pt"))
            print(f"  → New best dice: {best_dice:.4f}")
