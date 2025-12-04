"""Inference helpers for the PyTorch SplitterNet model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import torch

from .model import SplitterNetTorch
from .data import _load_image


def load_model(checkpoint_path: str, device: str) -> SplitterNetTorch:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SplitterNetTorch()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def denoise_image(model: SplitterNetTorch, image_path: str, device: str) -> torch.Tensor:
    tensor = _load_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    return output.squeeze(0).cpu()


def save_image(tensor: torch.Tensor, output_path: str) -> None:
    clipped = torch.clamp(tensor, 0.0, 1.0)
    array = (clipped.permute(1, 2, 0).numpy() * 65535.0).astype("uint16")
    imageio.imwrite(output_path, array)


def _iter_images(path: str) -> Iterable[str]:
    path_obj = Path(path)
    if path_obj.is_file():
        yield str(path_obj)
    else:
        for file in sorted(path_obj.iterdir()):
            if file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                yield str(file)


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Run inference with the PyTorch SplitterNet")
    parser.add_argument("checkpoint", help="Path to a training checkpoint")
    parser.add_argument("input_path", help="File or directory to denoise")
    parser.add_argument("output_dir", help="Directory to store the denoised outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Execution device")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - CLI
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.checkpoint, args.device)

    for image_path in _iter_images(args.input_path):
        output = denoise_image(model, image_path, args.device)
        out_path = Path(args.output_dir) / Path(image_path).name
        save_image(output, str(out_path))
        print(f"Saved {out_path}")
