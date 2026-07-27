#!/usr/bin/env python3
"""
Fetch the sherpa-onnx arm64-v8a JNI libraries into app/src/main/jniLibs/.

These are prebuilt Apache-2.0 binaries from the pinned sherpa-onnx release,
so they are not committed to this repository. setup_android.py calls this
automatically when the libraries are missing.

    python fetch_libs.py
"""

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SHERPA_VERSION = "1.13.3"
TARBALL = (
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"v{SHERPA_VERSION}/sherpa-onnx-v{SHERPA_VERSION}-android.tar.bz2"
)
ABI = "arm64-v8a"

HERE = Path(__file__).resolve().parent
DEST = HERE / "app" / "src" / "main" / "jniLibs" / ABI


def libs_present() -> bool:
    return (DEST / "libsherpa-onnx-jni.so").exists() and \
           (DEST / "libonnxruntime.so").exists()


def download(url: str, dest: Path) -> None:
    def hook(blocks, block_size, total):
        done = blocks * block_size
        pct = min(100.0, done * 100.0 / total) if total > 0 else 0
        sys.stdout.write(f"\r  {done / 1e6:8.1f} MB  ({pct:5.1f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def main() -> None:
    if libs_present():
        print(f"JNI libraries already present in jniLibs/{ABI}, nothing to do.")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tar_path = tmp / "android.tar.bz2"
        print(f"Downloading sherpa-onnx v{SHERPA_VERSION} Android libraries")
        download(TARBALL, tar_path)

        print("Extracting ...")
        with tarfile.open(tar_path, "r:bz2") as tf:
            members = [m for m in tf.getmembers()
                       if f"/{ABI}/" in m.name and m.name.endswith(".so")]
            if not members:
                sys.exit(f"No {ABI} .so files found in the archive.")
            for m in members:
                extracted = tf.extractfile(m)
                if extracted is None:
                    continue
                out = DEST / Path(m.name).name
                with open(out, "wb") as fh:
                    shutil.copyfileobj(extracted, fh)
                print(f"  {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")

    print(f"\nDone. Libraries in app/src/main/jniLibs/{ABI}/")


if __name__ == "__main__":
    main()
