from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
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


def _build_scheduler(optimizer, cfg: SteelConfig):
    """Linear warmup then cosine annealing."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.num_epochs - cfg.warmup_epochs, 1),
        eta_min=cfg.learning_rate / 100,
    )
    if cfg.warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
    )


def train(args: argparse.Namespace) -> None:
    data_dir = args.data_dir

    if getattr(args, "preset", "default") == "h200":
        cfg = SteelConfig.h200()
    else:
        cfg = SteelConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            compile_model=getattr(args, "compile", False),
            amp_dtype="bfloat16" if getattr(args, "bfloat16", False) else "float16",
        )

    # Paths always follow CLI --data-dir
    cfg.data_dir = data_dir
    cfg.train_csv = os.path.join(data_dir, "train.csv")
    cfg.image_dir = os.path.join(data_dir, "train_images")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_scaler = amp_dtype == torch.float16  # bfloat16 doesn't need a GradScaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device}  amp={cfg.amp_dtype}  compile={cfg.compile_model}")
    print(f"image={cfg.image_height}×{cfg.image_width}  "
          f"embed_dim={cfg.embed_dim}  depth={cfg.depth}  batch={cfg.batch_size}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    df = load_train_df(cfg.train_csv)
    pivot = build_pivot(df)

    n_val = max(1, int(len(pivot) * cfg.val_split))
    train_pivot = pivot.iloc[: len(pivot) - n_val]
    val_pivot = pivot.iloc[len(pivot) - n_val :]

    train_tf = get_transforms(cfg.image_height, cfg.image_width, phase="train")
    val_tf = get_transforms(cfg.image_height, cfg.image_width, phase="val")

    train_ds = SteelDataset(train_pivot, cfg.image_dir, train_tf)
    val_ds = SteelDataset(val_pivot, cfg.image_dir, val_tf)

    train_loader = _make_loader(train_ds, cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader = _make_loader(val_ds, cfg.batch_size, cfg.num_workers, shuffle=False)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    vit_cfg = cfg.vit_config()
    model = ViTSegmenter(vit_cfg, num_seg_classes=4).to(device)

    if getattr(args, "resume", None):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    if cfg.compile_model:
        print("Compiling model with torch.compile (max-autotune) …")
        model = torch.compile(model, mode="max-autotune")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    # ------------------------------------------------------------------
    # Optimiser, scheduler, loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
    )
    scheduler = _build_scheduler(optimizer, cfg)
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_dice = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [train]", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, masks)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= len(train_loader)

        # ---- validate ----
        model.eval()
        val_loss = val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [val]", leave=False):
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(images)
                    loss = criterion(logits, masks)
                val_loss += loss.item()
                val_dice += dice_score(logits, masks, threshold=cfg.threshold)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs} | "
            f"lr={lr_now:.2e} | "
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
