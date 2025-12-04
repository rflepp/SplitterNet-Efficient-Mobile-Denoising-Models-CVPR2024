"""PyTorch data loading helpers mirroring the TensorFlow pipeline."""

from __future__ import annotations

import os
import random
import re
from glob import glob
from typing import Iterable, List, Sequence, Tuple

import imageio.v2 as imageio
import torch
from torch.utils.data import DataLoader, Dataset


def _split_train_val(items: Sequence[str], val_split: float) -> Tuple[List[str], List[str]]:
    split_index = int(len(items) * (1 - val_split))
    return list(items[:split_index]), list(items[split_index:])


class PatchDataset(Dataset):
    """Dataset that loads noisy/denoised image patches from disk."""

    def __init__(self, noisy: Sequence[str], clean: Sequence[str]) -> None:
        assert len(noisy) == len(clean), "Noisy and clean datasets must align"
        self.noisy = list(noisy)
        self.clean = list(clean)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.noisy)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_img = _load_image(self.noisy[idx])
        clean_img = _load_image(self.clean[idx])
        # Deterministic paired flips keep inputs/targets aligned
        if random.random() < 0.5:
            noisy_img = torch.flip(noisy_img, dims=[1])
            clean_img = torch.flip(clean_img, dims=[1])
        if random.random() < 0.5:
            noisy_img = torch.flip(noisy_img, dims=[2])
            clean_img = torch.flip(clean_img, dims=[2])
        return noisy_img, clean_img


def build_dataloaders(
    original_images_path: str,
    denoised_images_path: str,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    noisy_paths, clean_paths = _collect_and_pair_images(original_images_path, denoised_images_path)

    seed = 2345
    random.seed(seed)
    random.shuffle(noisy_paths)
    random.seed(seed)
    random.shuffle(clean_paths)

    noisy_train, noisy_val = _split_train_val(noisy_paths, val_split)
    clean_train, clean_val = _split_train_val(clean_paths, val_split)

    train_dataset = PatchDataset(noisy_train, clean_train)
    val_dataset = PatchDataset(noisy_val, clean_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def _collect_and_pair_images(original_images_path: str, denoised_images_path: str) -> Tuple[List[str], List[str]]:
    original_images: List[Iterable[str]] = []
    denoised_images: List[Iterable[str]] = []

    for directory in os.listdir(original_images_path):
        if original_images_path == "/your/path/patches/":  # You need to define the folder of your one to one MIDD
            originals, denoised = pair_images_simple(
                os.path.join(original_images_path, directory, "original_patches"),
                os.path.join(denoised_images_path, directory, "denoised_patches"),
            )
        elif original_images_path == "/your/path/one/to/20/dataset":  # You need to define the folder of your full MIDD (20 noisy to 1 GT)
            originals, denoised = pair_images(
                os.path.join(original_images_path, directory, "original_20_patches"),
                os.path.join(denoised_images_path, directory, "denoised_patches"),
            )
        else:
            original_images = sorted(glob(os.path.join(original_images_path, "*")))
            denoised_images = sorted(glob(os.path.join(denoised_images_path, "*")))
            break
        original_images.append(originals)
        denoised_images.append(denoised)

    flat_original = _flatten_if_list_of_lists(original_images)
    flat_denoised = _flatten_if_list_of_lists(denoised_images)

    index, values = compare_filenames(flat_original, flat_denoised)
    if index is not None:
        raise ValueError(
            f"The noisy and ground truth image lists diverge at index {index}: {values[0]} vs {values[1]}"
        )
    return flat_original, flat_denoised


def pair_images(folder_path_original: str, folder_path_denoised: str) -> Tuple[List[str], List[str]]:
    original_images_out: List[str] = []
    ground_truth_images: List[Tuple[str, str]] = []
    ground_truth_images_out: List[str] = []
    file_dict: dict[str, List[str]] = {}

    for file_name in os.listdir(folder_path_original):
        if "image" in file_name:
            base_name = "_".join(file_name.split("_")[:-4])
            base_name2 = file_name.split("_")[-1][:-4]
            base_name1 = file_name.split("_")[-2]
            base_name = base_name + "_" + base_name1 + "_" + base_name2
            full_path = os.path.join(folder_path_original, file_name)
            file_dict.setdefault(base_name, []).append(full_path)

    for file_name in os.listdir(folder_path_denoised):
        base_name = _strip_extension(file_name)
        if base_name is None:
            continue
        full_path = os.path.join(folder_path_denoised, file_name)
        ground_truth_images.append((base_name, full_path))

    for gt_base_name, gt_path in ground_truth_images:
        if gt_base_name in file_dict:
            for original_path in file_dict[gt_base_name]:
                original_images_out.append(original_path)
                ground_truth_images_out.append(gt_path)

    return original_images_out, ground_truth_images_out


def pair_images_simple(folder_path_original: str, folder_path_denoised: str) -> Tuple[List[str], List[str]]:
    original_images_out: List[str] = []
    ground_truth_images: List[Tuple[str, str]] = []
    ground_truth_images_out: List[str] = []
    file_dict: dict[str, List[str]] = {}

    for file_name in os.listdir(folder_path_original):
        base_name = _strip_extension(file_name)
        if base_name is None:
            continue
        full_path = os.path.join(folder_path_original, file_name)
        file_dict.setdefault(base_name, []).append(full_path)

    for file_name in os.listdir(folder_path_original):
        if "image" in file_name:
            base_name = "_".join(file_name.split("_")[:-4])
            base_name2 = file_name.split("_")[-1][:-4]
            base_name1 = file_name.split("_")[-2]
            base_name = base_name + "_" + base_name1 + "_" + base_name2
            full_path = os.path.join(folder_path_original, file_name)
            file_dict.setdefault(base_name, []).append(full_path)

    for file_name in os.listdir(folder_path_denoised):
        base_name = _strip_extension(file_name)
        if base_name is None:
            continue
        full_path = os.path.join(folder_path_denoised, file_name)
        ground_truth_images.append((base_name, full_path))

    for gt_base_name, gt_path in ground_truth_images:
        if gt_base_name in file_dict:
            for original_path in file_dict[gt_base_name]:
                original_images_out.append(original_path)
                ground_truth_images_out.append(gt_path)

    return original_images_out, ground_truth_images_out


def compare_filenames(list1: Sequence[str], list2: Sequence[str]):
    for index, (path1, path2) in enumerate(zip(list1, list2)):
        filename1 = os.path.splitext(os.path.basename(path1))[0]
        filename2 = os.path.splitext(os.path.basename(path2))[0]
        filename1_matched = remove_file_x_pattern(filename1)
        if filename1_matched != filename2:
            return index, (filename1, filename2, path1, path2)
    return None, None


def remove_file_x_pattern(text: str) -> str:
    pattern = r"_file_(1?[0-9])"
    return re.sub(pattern, "", text)


def _strip_extension(file_name: str) -> str | None:
    lowered = file_name.lower()
    if lowered.endswith((".jpg", ".jpeg", ".png")):
        return os.path.splitext(file_name)[0]
    return None


def _flatten_if_list_of_lists(lst: Iterable[Iterable[str]]) -> List[str]:
    if all(isinstance(sublist, list) for sublist in lst):
        return [item for sublist in lst for item in sublist]
    return list(lst)


def _load_image(image_path: str, normalization_factor: float = 65535.0) -> torch.Tensor:
    image = imageio.imread(image_path)
    if image.dtype != "float32":
        image = image.astype("float32")
    image = torch.from_numpy(image) / normalization_factor
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    # HWC -> CHW
    return image.permute(2, 0, 1)
