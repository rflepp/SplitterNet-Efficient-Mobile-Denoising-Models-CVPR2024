"""Crop full-size images into non-overlapping square patches.

Example (run once for the noisy and once for the ground-truth images):
    python data_preprocessing/cropping_parallel.py dataset/uncropped/original dataset/patches/original_patches
    python data_preprocessing/cropping_parallel.py dataset/uncropped/denoised dataset/patches/denoised_patches
"""
import argparse
import os

import cv2
from joblib import Parallel, delayed


def create_patches(image_path, output_dir, size=256):
    """Write every full ``size`` x ``size`` patch of an image as a JPEG (quality 100)."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Could not read {image_path}")
        return

    stem = os.path.splitext(os.path.basename(image_path))[0]
    height, width = image.shape[:2]
    for i in range(width // size):
        for j in range(height // size):
            patch = image[j * size:(j + 1) * size, i * size:(i + 1) * size]
            out_path = os.path.join(output_dir, f"image_{stem}_{i}_{j}.jpg")
            cv2.imwrite(out_path, patch, [cv2.IMWRITE_JPEG_QUALITY, 100])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--size", type=int, default=256, help="Patch size in pixels (default: 256)")
    parser.add_argument("--jobs", type=int, default=10, help="Number of parallel workers (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir))
    Parallel(n_jobs=args.jobs)(delayed(create_patches)(p, args.output_dir, args.size) for p in paths)


if __name__ == "__main__":
    main()
