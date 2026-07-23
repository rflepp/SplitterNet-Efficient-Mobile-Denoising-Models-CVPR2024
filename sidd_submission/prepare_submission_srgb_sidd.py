import os
import sys
import wget
import scipy.io
import numpy as np
import tensorflow as tf

# Add parent directory to sys.path to allow importing models from anywhere
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

from models import SplitterNet


def resolve_path(relative_path):
    """Resolves path relative to current working dir or project root."""
    if os.path.exists(relative_path):
        return relative_path
    path_in_parent = os.path.join(parent_dir, relative_path)
    if os.path.exists(path_in_parent):
        return path_in_parent
    return relative_path


# Load SplitterNet model with model weights
weights_path = resolve_path("model_weights/SplitterNet_MIDD_model.h5")
print(f"Loading SplitterNet model with weights from: {weights_path}", flush=True)

try:
    model = keras.models.load_model(weights_path, compile=False)
    print("Successfully loaded SplitterNet model.", flush=True)
except Exception as e:
    print(f"Direct load failed ({e}). Constructing SplitterNet architecture and loading weights...", flush=True)
    model = SplitterNet.DYNUnet(input_shape=(None, None, 3), num_filters=32)
    model.load_weights(weights_path)
    print("Successfully loaded SplitterNet weights.", flush=True)


def my_srgb_denoiser(x):
    """sRGB denoiser using SplitterNet. Supports single patch (256, 256, 3) or block batch (N, 256, 256, 3)."""
    is_single = (x.ndim == 3)
    if is_single:
        x = np.expand_dims(x, axis=0)

    input_patch = x.astype(np.float32) / 255.0
    output_patch = model.predict(input_patch, batch_size=len(input_patch), verbose=0)

    if isinstance(output_patch, (list, tuple)):
        output_patch = output_patch[0]

    output_uint8 = np.clip(np.round(output_patch * 255.0), 0, 255).astype(np.uint8)
    return output_uint8[0] if is_single else output_uint8


# Download input file, if needed.
url = 'https://competitions.codalab.org/my/datasets/download/0d8a1e68-155d-4301-a8cd-9b829030d719'
input_file = resolve_path('BenchmarkNoisyBlocksSrgb.mat')
if os.path.exists(input_file):
    print(f'{input_file} exists. No need to download it.', flush=True)
else:
    print('Downloading input file BenchmarkNoisyBlocksSrgb.mat...', flush=True)
    wget.download(url, input_file)
    print('\nDownloaded successfully.', flush=True)

# Read inputs.
key = 'BenchmarkNoisyBlocksSrgb'
inputs = scipy.io.loadmat(input_file)
inputs = inputs[key]
print(f'inputs.shape = {inputs.shape}', flush=True)

# Denoising.
outputs = inputs.copy()
for i in range(inputs.shape[0]):
    print(f'Processing image {i + 1}/{inputs.shape[0]}...', flush=True)
    outputs[i, :, :, :, :] = my_srgb_denoiser(inputs[i, :, :, :, :])

print(f'outputs.shape = {outputs.shape}', flush=True)

# Save outputs to .mat file using exact CamelCase name and key.
output_file = 'SubmitSrgb.mat'

# Remove existing files to enforce exact CamelCase filename on Windows NTFS
for existing_file in ['submitsrgb.mat', output_file]:
    if os.path.exists(existing_file):
        try:
            os.remove(existing_file)
        except Exception:
            pass

print(f'Saving outputs to {output_file}', flush=True)
scipy.io.savemat(output_file, {'SubmitSrgb': outputs})

# TODO: Submit the output file SubmitSrgb.mat at 
# http://130.63.97.225/sidd/benchmark_submit.php

print('Done.', flush=True)
