"""Evaluate a denoising model (PSNR / SSIM) on a test set.

Example:
    python evaluate.py model_weights/SplitterNet_MIDD_model.h5 path/to/test_set --model SplitterNet
"""
import argparse
import logging

import keras
import numpy as np
import tensorflow as tf

import utils  # noqa: F401  (registers custom losses/metrics for model loading)
from dataloader import find_image_pairs
from models import MODEL_NAMES, build_model
from utils import PSNR

logger = logging.getLogger(__name__)


def load_model(path, model_name=None, num_filters=32):
    """Load a ``.keras`` model, or build ``model_name`` and load ``.h5``/``.weights.h5`` weights into it."""
    if path.endswith(".keras"):
        return keras.models.load_model(path, compile=False)
    if model_name is None:
        raise ValueError(f"{path} contains weights only: pass the architecture with --model")
    model = build_model(model_name, num_filters=num_filters)
    model.load_weights(path)
    return model


def read_image(path):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.cast(image, tf.float32).numpy() / 255.0


def ssim(a, b):
    return float(tf.image.ssim(a * 255.0, b * 255.0, max_val=255.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03))


def evaluate_model(model, test_dir):
    """Return the mean PSNR and SSIM of the model's outputs on all image pairs in ``test_dir``."""
    noisy_paths, clean_paths = find_image_pairs(test_dir)
    logger.info("Evaluating on %d images", len(noisy_paths))

    results = []
    for noisy_path, clean_path in zip(noisy_paths, clean_paths):
        if "blurry" in noisy_path:
            continue
        try:
            clean, noisy = read_image(clean_path), read_image(noisy_path)
            denoised = np.asarray(model(noisy[None], training=False))[0]
        except Exception as e:  # e.g. image size not supported by the architecture
            logger.warning("Skipping %s: %s", noisy_path, e)
            continue

        metrics = (PSNR(clean * 255, noisy * 255), PSNR(clean * 255, denoised * 255), ssim(clean, noisy), ssim(clean, denoised))
        results.append(metrics)
        logger.debug("%s: PSNR %.2f -> %.2f, SSIM %.4f -> %.4f", noisy_path, *metrics)

    if not results:
        raise RuntimeError(f"No images could be evaluated in {test_dir}")
    psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised = np.mean(results, axis=0)
    logger.info("PSNR: noisy %.3f, denoised %.3f", psnr_noisy, psnr_denoised)
    logger.info("SSIM: noisy %.4f, denoised %.4f", ssim_noisy, ssim_denoised)
    return psnr_denoised, ssim_denoised


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Trained .keras model, or a .h5 weights file together with --model")
    parser.add_argument("test_dir", help="Test set directory (see dataloader.py for supported layouts)")
    parser.add_argument("--model", choices=MODEL_NAMES, help="Architecture to build when model_path contains only weights")
    parser.add_argument("--filter-exp", type=int, default=5, help="Number of filters = 2**filter_exp (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log per-image metrics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = load_model(args.model_path, args.model, 2 ** args.filter_exp)
    evaluate_model(model, args.test_dir)


if __name__ == "__main__":
    main()
