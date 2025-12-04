"""Training entrypoint for the PyTorch SplitterNet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import build_dataloaders
from .model import SplitterNetTorch


def _psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = nn.functional.mse_loss(pred, target, reduction="mean")
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def train(
    dataset_path: str,
    batch_size: int,
    epochs: int,
    num_filters: int,
    val_split: float,
    lr: float,
    num_workers: int,
    device: str,
    save_dir: str,
    checkpoint_path: str | None,
) -> Path:
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_root / "logs"))

    train_loader, val_loader = build_dataloaders(dataset_path, dataset_path, batch_size, val_split, num_workers)

    model = SplitterNetTorch(num_filters=num_filters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    start_epoch = 0
    best_psnr = -1.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        best_psnr = checkpoint.get("best_psnr", best_psnr)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * noisy.size(0)
        epoch_loss /= len(train_loader.dataset)
        writer.add_scalar("train/loss", epoch_loss, epoch)

        val_psnr = _evaluate(model, val_loader, device)
        writer.add_scalar("val/psnr", val_psnr, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": max(best_psnr, val_psnr),
        }
        torch.save(checkpoint, save_root / "checkpoint_last.pt")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(checkpoint, save_root / "checkpoint_best.pt")

        print(f"Epoch {epoch+1}: loss={epoch_loss:.5f} val_psnr={val_psnr:.3f}dB")

    return save_root / "checkpoint_best.pt"


def _evaluate(model: SplitterNetTorch, loader: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    psnr_total = 0.0
    count = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            output = model(noisy)
            psnr_total += _psnr(output, clean).item() * noisy.size(0)
            count += noisy.size(0)
    return psnr_total / max(count, 1)


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Train the PyTorch SplitterNet denoiser")
    parser.add_argument("dataset_path", help="Path to paired noisy/clean patches")
    parser.add_argument("epochs", type=int, help="Number of epochs to train")
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("save_dir", help="Directory to store checkpoints and logs")
    parser.add_argument("--num_filters", type=int, default=32, help="Number of base convolution filters")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split from dataset")
    parser.add_argument("--lr", type=float, default=4e-4, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--checkpoint", help="Path to an existing checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - CLI
    args = parse_args()
    train(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_filters=args.num_filters,
        val_split=args.val_split,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        save_dir=args.save_dir,
        checkpoint_path=args.checkpoint,
    )
