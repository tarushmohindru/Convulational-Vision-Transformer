import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Severstal Steel Defect Detection — ViT segmentation pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --------------------------------------------------------------- download
    dp = sub.add_parser("download", help="Download competition data from Kaggle")
    dp.add_argument("--data-dir", default="data", help="Directory to save data into")
    dp.add_argument("--force", action="store_true", help="Re-download even if data exists")

    # ------------------------------------------------------------------ train
    tp = sub.add_parser("train", help="Train the ViT segmenter")
    tp.add_argument("--data-dir", default="data", help="Root data directory")
    tp.add_argument("--preset", choices=["default", "h200"], default="default",
                    help="Hyperparameter preset (h200: ViT-L, 1024-wide, bf16, compiled)")
    tp.add_argument("--epochs", type=int, default=30)
    tp.add_argument("--batch-size", type=int, default=8)
    tp.add_argument("--lr", type=float, default=1e-4)
    tp.add_argument("--compile", action="store_true", help="torch.compile the model (faster on H100/H200)")
    tp.add_argument("--bfloat16", action="store_true", help="Use bfloat16 AMP instead of float16")
    tp.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    # ------------------------------------------------------------------ infer
    ip = sub.add_parser("infer", help="Run inference and produce submission CSV")
    ip.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ip.add_argument("--data-dir", default="data/test_images", help="Test images directory")
    ip.add_argument("--threshold", type=float, default=0.5)
    ip.add_argument("--output", default="submission.csv", help="Output CSV path")

    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

    if args.command == "download":
        from src.download import download
        download(args)
    elif args.command == "train":
        from src.train import train
        train(args)
    elif args.command == "infer":
        from src.inference import infer
        infer(args)


if __name__ == "__main__":
    main()
