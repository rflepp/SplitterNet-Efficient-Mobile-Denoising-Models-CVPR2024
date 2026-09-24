"""Convert a denoising model to TensorFlow Lite for on-device benchmarking.

Examples:
    python converter.py --model SplitterNet --weights model_weights/SplitterNet_MIDD_model.h5
    python converter.py --model SplitterNet_LN --height 720 --width 480 --output splitternet_ln.tflite

The resulting .tflite file can be benchmarked with the PRO mode of the AI Benchmark app
(https://ai-benchmark.com/workshops/mai/2021/#runtime).
"""
import argparse
import os
import tempfile

import tensorflow as tf

from evaluate import load_model
from models import MODEL_NAMES, build_model


def convert(model, output_path, optimize=False):
    """Export ``model`` as a TFLite flatbuffer and return its size in bytes."""
    with tempfile.TemporaryDirectory() as saved_model_dir:
        model.export(saved_model_dir, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return len(tflite_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="SplitterNet", choices=MODEL_NAMES)
    parser.add_argument("--weights", help="Optional .keras model or .h5 weights to convert")
    parser.add_argument("--filter-exp", type=int, default=5, help="Number of filters = 2**filter_exp (default: 5)")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--optimize", action="store_true", help="Apply default TFLite optimisations (dynamic-range quantisation)")
    parser.add_argument("--output", help="Output path (default: <model>_<height>x<width>.tflite)")
    args = parser.parse_args()

    num_filters = 2 ** args.filter_exp
    input_shape = (args.height, args.width, 3)
    model = build_model(args.model, input_shape=input_shape, num_filters=num_filters)
    if args.weights:
        # Weights are loaded into a variable-size model and copied into the fixed-size one.
        model.set_weights(load_model(args.weights, args.model, num_filters).get_weights())

    output = args.output or f"{args.model.lower()}_{args.height}x{args.width}.tflite"
    size = convert(model, output, args.optimize)
    print(f"Saved {output} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
