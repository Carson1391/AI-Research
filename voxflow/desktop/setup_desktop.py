#!/usr/bin/env python3
"""
VoxFlow desktop setup.
Installs Python dependencies and downloads the ASR model.

Usage:
    python setup_desktop.py            (default: Nemotron 3.5 multilingual, 560 ms)
    python setup_desktop.py --model 2  (Nemotron English-only 0.6b, 560 ms)
    python setup_desktop.py --model 3  (small zipformer, quick smoke test)
"""

import argparse
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

MODELS = {
    "1": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
    "2": "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
    "3": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
}

PIP_PACKAGES = [
    "sherpa-onnx>=1.13.3",
    "sounddevice",
    "numpy",
    "keyboard",
    "pyperclip",
    "requests",
    "rapidfuzz",
]


def install_deps() -> None:
    print("[1/2] Installing Python packages ...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + PIP_PACKAGES
    subprocess.check_call(cmd)
    print("      done.\n")


def download_model(key: str) -> None:
    name = MODELS[key]
    dest = MODELS_DIR / name
    if (dest / "tokens.txt").exists():
        print(f"[2/2] Model already present: {dest}")
        return

    MODELS_DIR.mkdir(exist_ok=True)
    url = f"{RELEASE}/{name}.tar.bz2"
    tar_path = MODELS_DIR / f"{name}.tar.bz2"

    print(f"[2/2] Downloading {name}")
    print(f"      {url}")

    def hook(blocks, block_size, total):
        done = blocks * block_size
        mb = done / 1e6
        pct = min(100.0, done * 100.0 / total) if total > 0 else 0
        sys.stdout.write(f"\r      {mb:8.1f} MB  ({pct:5.1f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tar_path, reporthook=hook)
    print("\n      Extracting ...")
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(MODELS_DIR)
    tar_path.unlink()
    print(f"      Model ready: {dest}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1", choices=list(MODELS.keys()),
                    help="1 = Nemotron 3.5 multilingual (default), "
                         "2 = Nemotron English-only, 3 = small zipformer")
    args = ap.parse_args()

    install_deps()
    download_model(args.model)

    print("Setup complete. Run the app with:")
    print(f"    python {HERE / 'voxflow.py'}")
    if args.model != "1":
        print(f"\nYou chose model {args.model}. Set MODEL_DIR in voxflow.py to:")
        print(f'    "{MODELS[args.model]}"')


if __name__ == "__main__":
    main()
