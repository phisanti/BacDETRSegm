# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Channel adapter modules for converting grayscale/multi-channel images to pseudo-RGB.

Supported adapters:
- ResidualChannelAdapter: Lightweight residual blocks
- ConvNeXtChannelAdapter: ConvNeXtV2-inspired blocks
- GradientChannelAdapter: Pre-trained gradient flow prediction models (GradConvNeXt/UNeXt)
"""

from .residual import ResidualChannelAdapter
from .convnext import ConvNeXtChannelAdapter
from .gradient import GradientChannelAdapter
from typing import Optional
import torch.nn as nn


def build_channel_adapter(
    adapter_type: str,
    in_channels: int = 1,
    out_channels: int = 3,
    num_blocks: int = 2,
    intermediate_dim: int = 32,
    drop_path: float = 0.0,
    target_resolution: Optional[int] = None,
    model_name: Optional[str] = None,
    weights_path: Optional[str] = None,
    freeze: bool = False,
) -> nn.Module:
    """
    Factory function to build a channel adapter.

    Args:
        adapter_type: Type of adapter ("residual", "convnext", or "pretrained")
        in_channels: Number of input channels (e.g., 1 for grayscale)
        out_channels: Number of output channels (typically 3 for RGB)
        num_blocks: Number of blocks in the adapter (for residual/convnext)
        intermediate_dim: Hidden dimension (for residual/convnext)
        drop_path: Stochastic depth rate (for residual/convnext)
        target_resolution: Optional upsampling resolution (e.g., 312, 384, 432)
        model_name: Model name for pretrained adapter (e.g., "gradconvnext_atto")
        weights_path: Path to pre-trained weights (for pretrained adapter)
        freeze: Whether to freeze adapter parameters

    Returns:
        Channel adapter module

    Examples:
        # Residual adapter
        adapter = build_channel_adapter(
            adapter_type="residual",
            in_channels=1,
            out_channels=3,
            num_blocks=2,
            intermediate_dim=32,
            target_resolution=432
        )

        # ConvNeXt adapter
        adapter = build_channel_adapter(
            adapter_type="convnext",
            in_channels=1,
            out_channels=3,
            num_blocks=2,
            intermediate_dim=64,
            drop_path=0.1,
            target_resolution=432
        )

        # Gradient adapter (pre-trained)
        adapter = build_channel_adapter(
            adapter_type="gradient",
            model_name="gradconvnext_atto",
            weights_path="/path/to/weights.pth",
            target_resolution=432,
            freeze=True
        )
    """
    adapter_type = adapter_type.lower()

    if adapter_type == "residual":
        adapter = ResidualChannelAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            intermediate_dim=intermediate_dim,
            drop_path=drop_path,
            target_resolution=target_resolution,
        )

    elif adapter_type == "convnext":
        adapter = ConvNeXtChannelAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            intermediate_dim=intermediate_dim,
            drop_path=drop_path,
            target_resolution=target_resolution,
        )

    elif adapter_type == "pretrained" or adapter_type == "gradient":
        if model_name is None:
            raise ValueError("model_name must be specified for gradient/pretrained adapter")

        adapter = GradientChannelAdapter(
            model_name=model_name,
            weights_path=weights_path,
            target_resolution=target_resolution,
            freeze=freeze,
        )

    else:
        raise ValueError(
            f"Unknown adapter_type: {adapter_type}. "
            f"Supported types: 'residual', 'convnext', 'gradient', 'pretrained'"
        )

    # Freeze if requested (for non-gradient adapters)
    if freeze and adapter_type not in ["pretrained", "gradient"]:
        adapter.freeze()

    return adapter


__all__ = [
    "ResidualChannelAdapter",
    "ConvNeXtChannelAdapter",
    "GradientChannelAdapter",
    "build_channel_adapter",
]
