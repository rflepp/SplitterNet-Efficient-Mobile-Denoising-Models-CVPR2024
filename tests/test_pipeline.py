import os

import numpy as np
import tensorflow as tf

from dataloader import find_image_pairs, get_datasets, pairing_key
from evaluate import evaluate_model, load_model
from train import parse_blocks, train


def write_png(path, image, dtype=np.uint8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scale = np.iinfo(dtype).max
    tf.io.write_file(path, tf.io.encode_png(tf.constant((image * scale).astype(dtype))))


def make_flat_dataset(root, n=6, size=64, seed=0):
    rng = np.random.RandomState(seed)
    for i in range(n):
        clean = rng.rand(size, size, 3) * 0.5 + 0.25
        noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
        write_png(os.path.join(root, "original", f"{i}.png"), noisy)
        write_png(os.path.join(root, "denoised", f"{i}.png"), clean)


def test_pairing_key_ignores_burst_index():
    assert pairing_key("/a/image_1053_file_3_0_1.jpg") == "image_1053_0_1"
    assert pairing_key("image_1053_0_1.png") == "image_1053_0_1"


def test_find_pairs_in_scene_layout(tmp_path):
    img = np.zeros((8, 8, 3))
    for scene in ("s1", "s2"):
        for burst in range(3):
            write_png(str(tmp_path / scene / "original_20_patches" / f"image_{scene}_file_{burst}_0_0.png"), img)
        write_png(str(tmp_path / scene / "denoised_patches" / f"image_{scene}_0_0.png"), img)
    write_png(str(tmp_path / "s1" / "denoised_patches" / "image_unmatched_0_0.png"), img)

    noisy, clean = find_image_pairs(str(tmp_path))
    assert len(noisy) == len(clean) == 6
    assert all(pairing_key(n) == pairing_key(c) for n, c in zip(noisy, clean))


def test_datasets_apply_identical_flips(tmp_path):
    root = str(tmp_path)
    make_flat_dataset(root, n=10)
    train_ds, val_ds = get_datasets(root, batch_size=2, val_split=0.2)
    assert train_ds.cardinality() == 4 and val_ds.cardinality() == 1  # 8 train / 2 val images
    for noisy, clean in train_ds.unbatch():
        # Flipping only one of the two images would make them differ by far more than the noise.
        assert float(tf.reduce_mean(tf.abs(noisy - clean))) < 0.1


def test_parse_blocks():
    assert parse_blocks("[2,2,4,8]") == parse_blocks("2,2,4,8") == [2, 2, 4, 8]


def test_train_and_evaluate_end_to_end(tmp_path):
    data, out = str(tmp_path / "data"), str(tmp_path / "run")
    make_flat_dataset(data, n=6)
    train("SplitterNet", out, epochs=1, batch_size=2, dataset=data, test_dir=data, filter_exp=3)

    model_path = os.path.join(out, "trained_model.keras")
    assert os.path.exists(model_path)
    assert os.path.exists(os.path.join(out, "checkpoints", "model_01.keras"))

    psnr, ssim = evaluate_model(load_model(model_path), data)
    assert np.isfinite(psnr) and 0 < ssim <= 1

    # Resuming from the saved model works as well.
    train("SplitterNet", out, epochs=1, batch_size=2, dataset=data, filter_exp=3, checkpoint=model_path)
