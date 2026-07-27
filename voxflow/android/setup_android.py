#!/usr/bin/env python3
"""
VoxFlow Android setup.
Downloads the ASR model into app/src/main/assets/model/ with normalized
file names, and copies hotwords.txt into assets. Run this once BEFORE
building the APK in Android Studio.

Usage:
    python setup_android.py            (default: Nemotron 3.5 multilingual)
    python setup_android.py --model 2  (small zipformer, snappy on any phone)

Model 1 gives Wispr-level accuracy and 40 languages. The APK will be about
680 MB and the model takes several seconds to load on first keyboard open.
Runs well on modern arm64 phones (the sherpa-onnx project ships prebuilt
phone APKs with this exact model).

Model 2 is a 20M-parameter English zipformer: instant loading, tiny APK,
lower accuracy. Good for older phones or as a first test build.
"""

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "app" / "src" / "main" / "assets"
MODEL_DIR = ASSETS / "model"

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# name, kind, {target -> source file inside the package}
MODELS = {
    "1": (
        "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        "nemotron",
        {
            "encoder.onnx": "encoder.int8.onnx",
            "decoder.onnx": "decoder.int8.onnx",
            "joiner.onnx": "joiner.int8.onnx",
            "tokens.txt": "tokens.txt",
        },
    ),
    "2": (
        "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "zipformer",
        {
            "encoder.onnx": "encoder-epoch-99-avg-1.int8.onnx",
            # fp32 decoder: tiny file, int8 decoders degrade zipformer output
            "decoder.onnx": "decoder-epoch-99-avg-1.onnx",
            "joiner.onnx": "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt": "tokens.txt",
        },
    ),
}


def download(url: str, dest: Path) -> None:
    def hook(blocks, block_size, total):
        done = blocks * block_size
        pct = min(100.0, done * 100.0 / total) if total > 0 else 0
        sys.stdout.write(f"\r  {done / 1e6:8.1f} MB  ({pct:5.1f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1", choices=list(MODELS.keys()),
                    help="1 = Nemotron 3.5 multilingual (default), "
                         "2 = small English zipformer")
    ap.add_argument("--vocab-only", action="store_true",
                    help="only refresh assets/hotwords.txt, skip the model")
    args = ap.parse_args()

    if args.vocab_only:
        ASSETS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE.parent / "hotwords.txt", ASSETS / "hotwords.txt")
        print("hotwords.txt refreshed. Rebuild the APK to apply.")
        return

    # prebuilt JNI libraries are not committed to the repo, fetch on demand
    import fetch_libs
    if not fetch_libs.libs_present():
        fetch_libs.main()

    name, kind, mapping = MODELS[args.model]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tar_path = tmp / f"{name}.tar.bz2"
        print(f"Downloading {name}")
        download(f"{RELEASE}/{name}.tar.bz2", tar_path)

        print("Extracting ...")
        with tarfile.open(tar_path, "r:bz2") as tf:
            tf.extractall(tmp)

        src_dir = tmp / name
        for target, source in mapping.items():
            print(f"  {source}  ->  assets/model/{target}")
            shutil.copy2(src_dir / source, MODEL_DIR / target)

    (MODEL_DIR / "meta.txt").write_text(kind, encoding="utf-8")

    vocab_src = HERE.parent / "hotwords.txt"
    if vocab_src.exists():
        shutil.copy2(vocab_src, ASSETS / "hotwords.txt")
        print("  hotwords.txt -> assets/hotwords.txt")

    total_mb = sum(f.stat().st_size for f in MODEL_DIR.iterdir()) / 1e6
    print(f"\nDone. Model assets: {total_mb:.0f} MB ({kind}).")
    print("Now open the android/ folder in Android Studio and build:")
    print("  Build > Generate App Bundles or APKs > Generate APKs")
    print("or from a terminal in android/:  gradlew.bat assembleRelease")


if __name__ == "__main__":
    main()
