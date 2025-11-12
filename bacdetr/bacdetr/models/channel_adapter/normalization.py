# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Image normalization module for post-adapter processing.

This module provides normalization layers that can be placed after channel
adapters to normalize pseudo-RGB outputs before feeding to DinoV2 backbone.
"""

import torch
import torch.nn as nn


class ImageNormalization(nn.Module):
    """
    Normalize images with configurable mean/std statistics.

    Designed to be placed after channel adapters to normalize
    pseudo-RGB outputs before feeding to DinoV2 backbone.

    This layer applies standard normalization:
        output = (input - mean) / std

    The mean and std tensors are registered as buffers, so they will
    be moved to the correct device automatically with the model.

    Args:
        mean: Tuple of mean values for each channel (default: ImageNet/DinoV2 means)
        std: Tuple of std values for each channel (default: ImageNet/DinoV2 stds)

    Example:
        >>> norm = ImageNormalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        >>> x = torch.rand(2, 3, 224, 224)  # [B, C, H, W] in [0, 1]
        >>> normalized = norm(x)  # Normalized with DinoV2 statistics
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        # Convert to tensors and reshape for broadcasting: [1, C, 1, 1]
        mean_tensor = torch.tensor(mean).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std).view(1, -1, 1, 1)

        # Register as buffers so they move with the model to correct device
        self.register_buffer('mean', mean_tensor)
        self.register_buffer('std', std_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to input tensor.

        Args:
            x: Input tensor of shape [B, C, H, W] in range [0, 1]
               where C matches the number of channels in mean/std

        Returns:
            Normalized tensor with the same shape as input, normalized
            with the configured mean and std statistics

        Note:
            Input is expected to be in [0, 1] range (raw pixel values).
            After normalization, values will typically be in range ~[-2, +2.5]
            for DinoV2 statistics.
        """
        return (x - self.mean) / self.std

    def extra_repr(self) -> str:
        """String representation for print/debug."""
        mean_str = self.mean.squeeze().tolist()
        std_str = self.std.squeeze().tolist()
        return f'mean={mean_str}, std={std_str}'
