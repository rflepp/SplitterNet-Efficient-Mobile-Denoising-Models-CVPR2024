"""PyTorch port of the SplitterNet denoiser."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return self.activation(self.conv(x))


class MidBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.act = nn.LeakyReLU(inplace=True)
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        x = self.act(self.conv1(x))
        x = self.channel_att(x)
        x = x + residual
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        x = self.act(self.conv2(x))
        x = self.spatial_att(x)
        return x + residual


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(x, dim=(2, 3), keepdim=True)
        weights = self.conv(pooled)
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        pooled = F.pad(pooled, (1, 1, 1, 1), mode="reflect")
        attention = torch.sigmoid(self.conv(pooled))
        return x * attention


class DecoderBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels * 2, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        x = self.activation(self.deconv(x))
        return x + skip


class SplitterNetTorch(nn.Module):
    def __init__(self, in_channels: int = 3, num_filters: int = 32) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        # stage1: 2, stage2: 4, stage3: 8, stage4: 16
        for _ in range(30):
            self.down_blocks.append(DownsampleBlock(num_filters // 2, num_filters))

        self.mid_blocks = nn.ModuleList([MidBlock(num_filters) for _ in range(16)])

        self.decoder_blocks = nn.ModuleList([DecoderBlock(num_filters) for _ in range(15)])
        self.head = nn.Conv2d(num_filters, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = self.stem(x)
        skips.append(x)

        stage1, idx = self._downstage([x], start_idx=0, num_pairs=1)
        skips.extend(stage1)
        stage2, idx = self._downstage(stage1, start_idx=idx, num_pairs=2)
        skips.extend(stage2)
        stage3, idx = self._downstage(stage2, start_idx=idx, num_pairs=4)
        skips.extend(stage3)
        stage4, _ = self._downstage(stage3, start_idx=idx, num_pairs=8)

        mid = [block(t) for block, t in zip(self.mid_blocks, stage4)]

        up4 = self._decode_stage(mid, stage3, decoder_offset=0)
        up3 = self._decode_stage(up4, stage2, decoder_offset=8)
        up2 = self._decode_stage(up3, stage1, decoder_offset=12)
        up1 = self._decode_stage(up2, [skips[0]], decoder_offset=14)

        out = self.head(up1[0])
        return out + x

    def _downstage(self, inputs: List[torch.Tensor], start_idx: int, num_pairs: int) -> Tuple[List[torch.Tensor], int]:
        outputs: List[torch.Tensor] = []
        idx = start_idx
        for tensor in inputs:
            left, right = torch.chunk(tensor, 2, dim=1)
            outputs.append(self.down_blocks[idx](left))
            idx += 1
            outputs.append(self.down_blocks[idx](right))
            idx += 1
        assert len(outputs) == num_pairs * 2
        return outputs, idx

    def _decode_stage(
        self, tensors: List[torch.Tensor], skips: List[torch.Tensor], decoder_offset: int
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for i in range(0, len(tensors), 2):
            skip = skips[i // 2]
            block = self.decoder_blocks[decoder_offset + i // 2]
            outputs.append(block(tensors[i], tensors[i + 1], skip))
        return outputs
