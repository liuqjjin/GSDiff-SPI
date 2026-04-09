"""Lightweight 3D UNet for video diffusion score estimation.

Input / output: [B, 1, T, H, W]  (grayscale video, batch-first)
Sigma conditioning: log(sigma) -> sinusoidal embedding -> MLP -> scale+shift per block.

Architecture (32->64->128, ~3M params for 64x64x20):
  Conv3d_in  -> DownBlock(32->64) -> DownBlock(64->128)
  -> MiddleBlock(128) ->
  UpBlock(128->64) -> UpBlock(64->32) -> Conv3d_out
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ─── Sigma embedding ──────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """log(sigma) -> sinusoidal positional embedding [B, dim]."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: [B] -> [B, dim]."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=sigma.device, dtype=torch.float32) / half
        )
        args = sigma.float().unsqueeze(1).log() * freqs.unsqueeze(0)   # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=1)              # [B, dim]


class SigmaConditioner(nn.Module):
    """sigma -> (scale, shift) for FiLM conditioning."""

    def __init__(self, emb_dim: int, out_channels: int):
        super().__init__()
        self.emb = SinusoidalEmbedding(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, out_channels * 2),
        )

    def forward(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """sigma: [B] -> (scale [B,C,1,1,1], shift [B,C,1,1,1])."""
        h = self.mlp(self.emb(sigma))           # [B, 2*C]
        scale, shift = h.chunk(2, dim=1)        # [B, C] each
        return scale[:, :, None, None, None] + 1.0, shift[:, :, None, None, None]


# ─── Building blocks ──────────────────────────────────────────

class ResBlock3D(nn.Module):
    """Conv3d -> GroupNorm -> SiLU -> Conv3d -> GroupNorm -> SiLU + FiLM + skip."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn1   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.gn2   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act   = nn.SiLU()
        self.cond  = SigmaConditioner(emb_dim, out_ch)
        self.skip  = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        scale, shift = self.cond(sigma)
        h = h * scale + shift
        h = self.act(h)
        return h + self.skip(x)


class DownBlock3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 128):
        super().__init__()
        self.res = ResBlock3D(in_ch, out_ch, emb_dim)
        self.down = nn.AvgPool3d(2)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
        h = self.res(x, sigma)
        return self.down(h), h   # (downsampled, skip)


class UpBlock3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 128):
        super().__init__()
        # in_ch is after concat with skip, so actual input channels = in_ch + skip_ch
        self.res = ResBlock3D(in_ch, out_ch, emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, sigma: torch.Tensor):
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, sigma)


# ─── Full UNet ────────────────────────────────────────────────

class UNet3D(nn.Module):
    """Lightweight 3D UNet for video score estimation.

    Parameters
    ----------
    in_channels : int, default 1  (grayscale video)
    base_channels : int, default 32
    channel_mults : list[int], default [1, 2, 4]  -> [32, 64, 128]
    emb_dim : int, default 128  (sigma embedding dimension)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 channel_mults: Optional[List[int]] = None, emb_dim: int = 128):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4]
        chs = [base_channels * m for m in channel_mults]   # e.g. [32, 64, 128]

        # Input projection
        self.conv_in = nn.Conv3d(in_channels, chs[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.down_blocks.append(DownBlock3D(chs[i], chs[i + 1], emb_dim))

        # Middle
        self.mid = ResBlock3D(chs[-1], chs[-1], emb_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            # input channels = current + skip from encoder
            self.up_blocks.append(UpBlock3D(chs[i] + chs[i], chs[i - 1], emb_dim))

        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv3d(chs[0], in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, C, T, H, W]   noisy video
        sigma: [B]                noise level per sample
        Returns: [B, C, T, H, W] predicted noise epsilon
        """
        h = self.conv_in(x)

        skips = []
        for block in self.down_blocks:
            h, skip = block(h, sigma)
            skips.append(skip)

        h = self.mid(h, sigma)

        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, sigma)

        return self.conv_out(h)
