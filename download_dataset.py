"""
Download the real flower dataset (about 3,670 photos, 5 flower types, ~220 MB).

Source : TensorFlow "flower_photos" dataset (daisy, dandelion, roses, sunflowers, tulips)
Run    : python download_dataset.py
Result : data/flower_photos/<flower type>/*.jpg

No Kaggle account is needed for this one. If you prefer a Kaggle dataset, download
it from kaggle.com, unzip it into the data/ folder, and point the main script at it.
"""

import os
import sys
import tarfile
import urllib.request

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ARCHIVE = os.path.join(DATA_DIR, "flower_photos.tgz")
TARGET = os.path.join(DATA_DIR, "flower_photos")


def progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100, done * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {pct:5.1f}%  ({done / 1e6:.0f} / {total_size / 1e6:.0f} MB)")
        sys.stdout.flush()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isdir(TARGET):
        print(f"Dataset already exists at {TARGET}. Nothing to do.")
        return

    print(f"Downloading from {URL}")
    try:
        urllib.request.urlretrieve(URL, ARCHIVE, progress)
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("Check your internet connection, or download the dataset manually from")
        print("kaggle.com and unzip it into the 'data' folder.")
        if os.path.exists(ARCHIVE):
            os.remove(ARCHIVE)
        sys.exit(1)

    print("\nExtracting...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):      # safer extraction on newer Python
            tar.extractall(DATA_DIR, filter="data")
        else:
            tar.extractall(DATA_DIR)
    os.remove(ARCHIVE)

    classes = sorted(d for d in os.listdir(TARGET) if os.path.isdir(os.path.join(TARGET, d)))
    total = 0
    for c in classes:
        n = len([f for f in os.listdir(os.path.join(TARGET, c)) if f.lower().endswith(".jpg")])
        total += n
        print(f"  {c:<12} {n} images")
    print(f"Done: {total} images in {TARGET}")
    print("Now run:  python flower_image_classification.py data/flower_photos")


if __name__ == "__main__":
    main()
