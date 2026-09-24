"""Denoise the SIDD sRGB benchmark with the pretrained SplitterNet and write ``SubmitSrgb.mat``.

Submit the resulting file at http://130.63.97.225/sidd/benchmark_submit.php.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
import scipy.io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models import build_model  # noqa: E402

BENCHMARK_URL = "https://competitions.codalab.org/my/datasets/download/0d8a1e68-155d-4301-a8cd-9b829030d719"
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "model_weights", "SplitterNet_MIDD_model.h5")


def denoise(model, blocks):
    """Denoise a batch of uint8 sRGB blocks of shape (N, H, W, 3)."""
    output = model.predict(blocks.astype(np.float32) / 255.0, batch_size=len(blocks), verbose=0)
    return np.clip(np.round(output * 255.0), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--input", default="BenchmarkNoisyBlocksSrgb.mat", help="Downloaded automatically if missing")
    parser.add_argument("--output", default="SubmitSrgb.mat")
    args = parser.parse_args()

    print(f"Loading SplitterNet weights from {args.weights}", flush=True)
    model = build_model("SplitterNet", num_filters=32)
    model.load_weights(args.weights)

    if not os.path.exists(args.input):
        print(f"Downloading {args.input} ...", flush=True)
        urllib.request.urlretrieve(BENCHMARK_URL, args.input)

    inputs = scipy.io.loadmat(args.input)["BenchmarkNoisyBlocksSrgb"]
    print(f"inputs.shape = {inputs.shape}", flush=True)

    outputs = np.empty_like(inputs)
    for i, image_blocks in enumerate(inputs):
        print(f"Processing image {i + 1}/{len(inputs)}", flush=True)
        outputs[i] = denoise(model, image_blocks)

    # The benchmark expects this exact (case-sensitive) file name and key; remove stale
    # lower-case copies first so case-insensitive file systems keep the right name.
    if os.path.basename(args.output) == "SubmitSrgb.mat":
        stale = os.path.join(os.path.dirname(args.output), "submitsrgb.mat")
        if os.path.exists(stale):
            os.remove(stale)
    scipy.io.savemat(args.output, {"SubmitSrgb": outputs})
    print(f"Saved {args.output} (shape {outputs.shape})", flush=True)


if __name__ == "__main__":
    main()
