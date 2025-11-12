# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Lightweight residual-based channel adapter.
Converts grayscale (or multi-channel) images to pseudo-RGB for pre-trained backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Lightweight residual block with depthwise separable convolutions.

    Architecture:
        x -> Conv3x3 -> BN -> GELU -> Conv3x3 -> BN -> + -> out
        |_______________________________________________|
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_path(out)
        return out + identity


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
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ResidualChannelAdapter(nn.Module):
    """
    Lightweight residual channel adapter for grayscale -> pseudo-RGB conversion.

    Args:
        in_channels: Number of input channels (e.g., 1 for grayscale)
        out_channels: Number of output channels (typically 3 for RGB)
        num_blocks: Number of residual blocks (2-4 typical)
        intermediate_dim: Hidden dimension for processing (32-64 typical)
        drop_path: Stochastic depth rate (0.0-0.2 for regularization)
        target_resolution: Optional target resolution for upsampling (e.g., 312, 384, 432)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        num_blocks: int = 2,
        intermediate_dim: int = 32,
        drop_path: float = 0.0,
        target_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_resolution = target_resolution

        # Initial projection: in_channels -> intermediate_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_dim),
            nn.GELU(),
        )

        # Residual blocks for feature refinement
        drop_path_rates = [drop_path * (i / max(num_blocks - 1, 1)) for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            ResidualBlock(intermediate_dim, drop_path=drop_path_rates[i])
            for i in range(num_blocks)
        ])

        # Final projection: intermediate_dim -> out_channels
        self.head = nn.Sequential(
            nn.Conv2d(intermediate_dim, out_channels, kernel_size=1, bias=True),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
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
