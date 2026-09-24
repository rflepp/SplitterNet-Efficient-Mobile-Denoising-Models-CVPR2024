"""tf.data input pipeline and noisy/ground-truth image pairing.

Two dataset layouts are supported (see ``find_image_pairs``):

* flat:   ``<root>/original/*`` (noisy) and ``<root>/denoised/*`` (ground truth),
          paired by sorted file name.
* scenes: one sub-directory per scene, each with a noisy and a ground-truth folder
          (``original_patches``/``original_20_patches`` + ``denoised_patches`` as
          produced by ``data_preprocessing/cropping_parallel.py``, or
          ``test_set/original`` + ``test_set/denoised``). Files are paired by name,
          ignoring a ``_file_<N>`` burst index in the noisy file names, so one
          ground-truth image can be paired with several noisy captures.
"""
import logging
import os
import random
import re
from collections import defaultdict
from glob import glob

import tensorflow as tf

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
NOISY_DIRS = ("original_patches", "original_20_patches", os.path.join("test_set", "original"))
CLEAN_DIRS = ("denoised_patches", os.path.join("test_set", "denoised"))
SHUFFLE_SEED = 2345


def pairing_key(file_name):
    """Name used to match a noisy image to its ground truth (extension and burst index removed)."""
    stem, _ = os.path.splitext(os.path.basename(file_name))
    return re.sub(r"_file_\d+", "", stem)


def _images_in(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def pair_images(noisy_dir, clean_dir):
    """Pair every noisy image in ``noisy_dir`` with the ground truth of the same name in ``clean_dir``."""
    noisy_by_key = defaultdict(list)
    for path in _images_in(noisy_dir):
        noisy_by_key[pairing_key(path)].append(path)

    noisy, clean = [], []
    for clean_path in _images_in(clean_dir):
        for noisy_path in noisy_by_key.get(pairing_key(clean_path), []):
            noisy.append(noisy_path)
            clean.append(clean_path)
    return noisy, clean


def _first_existing(root, candidates):
    return next((os.path.join(root, c) for c in candidates if os.path.isdir(os.path.join(root, c))), None)


def find_image_pairs(root):
    """Return aligned lists of (noisy, ground truth) image paths below ``root``."""
    if os.path.isdir(os.path.join(root, "original")):
        noisy = sorted(glob(os.path.join(root, "original", "*")))
        clean = sorted(glob(os.path.join(root, "denoised", "*")))
        if len(noisy) != len(clean):
            raise ValueError(f"{root}: {len(noisy)} noisy but {len(clean)} ground-truth images")
        return noisy, clean

    noisy, clean = [], []
    for scene in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, scene)
        noisy_dir, clean_dir = _first_existing(scene_dir, NOISY_DIRS), _first_existing(scene_dir, CLEAN_DIRS)
        if noisy_dir is None or clean_dir is None:
            continue
        logger.info("Loading scene %s", scene)
        scene_noisy, scene_clean = pair_images(noisy_dir, clean_dir)
        noisy += scene_noisy
        clean += scene_clean

    if not noisy:
        raise ValueError(f"No image pairs found in {root}")
    return noisy, clean


def get_datasets(dataset_dir, batch_size, val_split=0.1):
    """Build shuffled training and validation datasets from ``dataset_dir``."""
    pairs = list(zip(*find_image_pairs(dataset_dir)))
    mismatches = [(n, c) for n, c in pairs if pairing_key(n) != pairing_key(c)]
    if mismatches:
        logger.warning("%d pairs have different names, e.g. %s <-> %s", len(mismatches), *mismatches[0])
    logger.info("Found %d image pairs", len(pairs))

    random.Random(SHUFFLE_SEED).shuffle(pairs)
    num_train = int(len(pairs) * (1 - val_split))
    return build_dataset(pairs[:num_train], batch_size, training=True), build_dataset(pairs[num_train:], batch_size)


def build_dataset(pairs, batch_size, training=False):
    """Load image pairs; training datasets are reshuffled every epoch, augmented and use full batches only."""
    noisy, clean = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(noisy), list(clean)))
    if training:
        dataset = dataset.shuffle(len(pairs), seed=SHUFFLE_SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda n, c: (read_image(n), read_image(c)), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(int(batch_size), drop_remainder=training).prefetch(tf.data.AUTOTUNE)


def augment_pair(noisy, clean):
    """Apply the same random flips to the noisy image and its ground truth."""
    stacked = tf.concat([noisy, clean], axis=-1)
    stacked = tf.image.random_flip_up_down(stacked)
    stacked = tf.image.random_flip_left_right(stacked)
    return stacked[..., :3], stacked[..., 3:]


def read_image(path):
    """Read an 8- or 16-bit RGB image as float32 in [0, 1]."""
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, dtype=tf.uint16, expand_animations=False)
    return tf.cast(image, tf.float32) / 65535.0
