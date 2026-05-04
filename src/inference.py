from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import SteelConfig
from dataset import (
    NUM_CLASSES,
    SteelTestDataset,
    encode_rle,
    get_transforms,
)
from segmentation import ViTSegmenter


def _postprocess(prob_map: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    """Threshold probability map and remove tiny blobs."""
    binary = (prob_map > threshold).astype(np.uint8)
    if binary.sum() < min_area:
        binary[:] = 0
    return binary


def infer(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg: SteelConfig = ckpt["cfg"]

    # Allow CLI overrides of threshold / output path
    threshold = getattr(args, "threshold", cfg.threshold)
    output_path = getattr(args, "output", cfg.submission_path)
    data_dir = getattr(args, "data_dir", cfg.test_image_dir)

    test_image_dir = data_dir if os.path.isdir(data_dir) else cfg.test_image_dir
    image_ids = sorted(os.listdir(test_image_dir))
    print(f"Found {len(image_ids)} test images in {test_image_dir}")

    transform = get_transforms(cfg.image_height, cfg.image_width, phase="val")
    dataset = SteelTestDataset(image_ids, test_image_dir, transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    vit_cfg = cfg.vit_config()
    model = ViTSegmenter(vit_cfg, num_seg_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows: list[dict] = []

    with torch.no_grad():
        for img_ids, images in tqdm(loader, desc="Inference"):
            images = images.to(device)
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 4, H, W)

            # Resize predictions back to original 256×1600
            import torch.nn.functional as F
            logits_full = F.interpolate(logits, size=(256, 1600), mode="bilinear", align_corners=False)
            probs_full = torch.sigmoid(logits_full).cpu().numpy()

            for b_idx, img_id in enumerate(img_ids):
                for cls_idx in range(NUM_CLASSES):
                    prob_map = probs_full[b_idx, cls_idx]
                    binary = _postprocess(prob_map, threshold, cfg.min_mask_area)
                    rle = encode_rle(binary)
                    rows.append({
                        "ImageId_ClassId": f"{img_id}_{cls_idx + 1}",
                        "EncodedPixels": rle,
                    })

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
