# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Lightweight ConvNeXtV2-inspired channel adapter.
Converts grayscale (or multi-channel) images to pseudo-RGB for pre-trained backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels-first 2D tensors.
    Used in ConvNeXt architecture.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    Lightweight ConvNeXtV2 block.

    Architecture (inspired by ConvNeXt):
        x -> DWConv7x7 -> LayerNorm -> Linear(4*dim) -> GELU -> Linear(dim) -> DropPath -> + -> out
        |_______________________________________________________________________________|

    Uses depthwise separable convolutions for efficiency.
    """
    def __init__(self, dim: int, drop_path: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        # Depthwise convolution (spatial mixing)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm2d(dim)

        # Pointwise MLP (channel mixing) - 1x1 convs act as linear layers
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return x + identity


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    Used for regularization in residual networks.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtChannelAdapter(nn.Module):
    """
    Lightweight ConvNeXtV2-inspired channel adapter for grayscale -> pseudo-RGB conversion.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale)
        out_channels: Number of output channels (typically 3 for RGB)
        num_blocks: Number of ConvNeXt blocks (2-4 typical)
        intermediate_dim: Hidden dimension for processing (32-64 typical)
        drop_path: Stochastic depth rate (0.0-0.2 for regularization)
        target_resolution: Optional target resolution for upsampling (e.g., 312, 384, 432)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        num_blocks: int = 2,
        intermediate_dim: int = 64,
        drop_path: float = 0.1,
        target_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_resolution = target_resolution

        # Stem: Initial projection with patchify-like operation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_dim, kernel_size=4, stride=1, padding=2, bias=False),
            LayerNorm2d(intermediate_dim),
        )

        # ConvNeXt blocks for feature refinement
        drop_path_rates = [drop_path * (i / max(num_blocks - 1, 1)) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(intermediate_dim, drop_path=drop_path_rates[i])
            for i in range(num_blocks)
        ])

        # Head: Final projection to output channels
        self.norm = LayerNorm2d(intermediate_dim)
        self.head = nn.Conv2d(intermediate_dim, out_channels, kernel_size=1, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H', W')
            where H', W' = target_resolution if specified, else H, W
        """
        # Channel conversion and feature extraction
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)

        # Optional upsampling to target resolution
        if self.target_resolution is not None:
            if x.shape[-2] != self.target_resolution or x.shape[-1] != self.target_resolution:
                x = F.interpolate(
                    x,
                    size=(self.target_resolution, self.target_resolution),
                    mode='bilinear',
                    align_corners=False
                )

        return x

    def freeze(self):
        """Freeze all parameters (useful for transfer learning)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
