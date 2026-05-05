from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

COMPETITION = "severstal-steel-defect-detection"
EXPECTED_FILES = ["train.csv", "train_images", "test_images", "sample_submission.csv"]


def _check_credentials() -> None:
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    has_file = os.path.exists(kaggle_json)
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if not has_file and not has_env:
        raise SystemExit(
            "Kaggle credentials not found.\n"
            "Either:\n"
            "  1. Place your API token at ~/.kaggle/kaggle.json\n"
            "     (download from https://www.kaggle.com/settings → API → Create New Token)\n"
            "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
    if has_file:
        os.chmod(kaggle_json, 0o600)


def _extract(zip_path: str, dest: str) -> None:
    print(f"Extracting {os.path.basename(zip_path)} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    os.remove(zip_path)


def download(args: argparse.Namespace) -> None:
    _check_credentials()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    already_present = [f for f in EXPECTED_FILES if os.path.exists(os.path.join(data_dir, f))]
    if already_present and not args.force:
        print(f"Data already present in '{data_dir}': {already_present}")
        print("Pass --force to re-download.")
        return

    # Use the kaggle CLI — stable across all package versions
    kaggle_bin = os.path.join(os.path.dirname(sys.executable), "kaggle")
    cmd = [
        kaggle_bin, "competitions", "download",
        "-c", COMPETITION,
        "-p", data_dir,
    ]
    print(f"Downloading '{COMPETITION}' into '{data_dir}' …")
    subprocess.run(cmd, check=True)

    # Extract any zips that were downloaded
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".zip"):
            _extract(os.path.join(data_dir, fname), data_dir)

    print("Download complete. Contents:")
    for item in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, item)
        size = (
            f"{os.path.getsize(full) / 1e6:.1f} MB"
            if os.path.isfile(full)
            else f"{sum(1 for _ in os.scandir(full))} files"
        )
        print(f"  {item}  ({size})")
