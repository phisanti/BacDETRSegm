# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Gradient-based channel adapter for pre-trained gradient flow prediction networks.
Loads GradientConvNext and GradientConvUNext models trained to predict gradient flows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pathlib import Path


# Model configuration registry
GRADIENT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # UNext variants (default choice)
    "gradconvunext_atto": {
        "cls": "GradientConvUNext",
        "kwargs": {"depths": [2, 2, 4, 2], "dims": [10, 20, 40, 80]},
    },
    "gradconvunext_femto": {
        "cls": "GradientConvUNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [16, 32, 64, 128]},
    },
    "gradconvunext_pico": {
        "cls": "GradientConvUNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [24, 48, 96, 192]},
    },
    "gradconvunext_nano": {
        "cls": "GradientConvUNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [32, 64, 128, 256]},
    },
    "gradconvunext_tiny": {
        "cls": "GradientConvUNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},
    },
    # Plain encoder-decoder variants
    "gradconvnext_atto": {
        "cls": "GradientConvNext",
        "kwargs": {"depths": [2, 2, 4, 2], "dims": [10, 20, 40, 80]},
    },
    "gradconvnext_femto": {
        "cls": "GradientConvNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [16, 32, 64, 128]},
    },
    "gradconvnext_pico": {
        "cls": "GradientConvNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [24, 48, 96, 192]},
    },
    "gradconvnext_nano": {
        "cls": "GradientConvNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [32, 64, 128, 256]},
    },
    "gradconvnext_tiny": {
        "cls": "GradientConvNext",
        "kwargs": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},
    },
}


class GradientChannelAdapter(nn.Module):
    """
    Gradient-based channel adapter using pre-trained gradient flow prediction networks.

    This adapter loads pre-trained GradientConvNext/UNext models that have been trained
    to predict gradient flows from grayscale images. The adapter outputs 3 channels:
    [source, dx, dy] where source is the original grayscale image and dx, dy are the
    predicted gradients. These are used as pseudo-RGB input for the backbone.

    This adapter is responsible ONLY for channel interpolation (1 channel -> 3 channels).
    Image resizing should be handled by the dataloader before the adapter.

    Args:
        model_name: Name of the pre-trained model (e.g., "gradconvnext_atto", "gradconvunext_femto")
        weights_path: Path to the pre-trained weights file
        freeze: Whether to freeze the model after loading (default: False)
        apply_scaling: Whether to apply linear scaling to gradients from [-1,1] to [0,1] (default: True)

    Output:
        3 channels: [source, dx, dy]
        - Channel 0: Original grayscale image [0, 1]
        - Channel 1: Horizontal gradient dx (scaled to [0, 1] if apply_scaling=True)
        - Channel 2: Vertical gradient dy (scaled to [0, 1] if apply_scaling=True)
    """
    def __init__(
        self,
        model_name: str,
        weights_path: Optional[str] = None,
        freeze: bool = False,
        apply_scaling: bool = True,
        target_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.weights_path = weights_path
        self.apply_scaling = apply_scaling
        self.target_resolution = target_resolution

        # Build the gradient model
        self.model = self._build_model(model_name)

        # Load pre-trained weights if provided
        if weights_path is not None:
            self._load_weights(weights_path)

        # Freeze if requested
        if freeze:
            self.freeze()

        # Infer input/output channels from the model
        self.in_channels = self._get_in_channels()
        self.out_channels = self._get_out_channels()

    def _build_model(self, model_name: str) -> nn.Module:
        """
        Build the gradient model based on model_name.

        Supports:
        - GradientConvNext models (gradconvnext_*)
        - GradientConvUNext models (gradconvunext_*)
        """
        model_name_lower = model_name.lower()

        # Check if model_name is in registry
        if model_name_lower not in GRADIENT_MODEL_CONFIGS:
            available_models = ", ".join(sorted(GRADIENT_MODEL_CONFIGS.keys()))
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available models: {available_models}"
            )

        config = GRADIENT_MODEL_CONFIGS[model_name_lower]
        model_cls_name = config["cls"]
        model_kwargs = config["kwargs"]

        try:
            # Import the appropriate class
            if model_cls_name == "GradientConvNext":
                from gcn.gradconvnext import GradientConvNext  # type: ignore
                model = GradientConvNext(**model_kwargs)
            elif model_cls_name == "GradientConvUNext":
                from gcn.gradconvunext import GradientConvUNext  # type: ignore
                model = GradientConvUNext(**model_kwargs)
            else:
                raise ValueError(f"Unknown model class: {model_cls_name}")

            print(f"Built gradient model: {model_name} with config {model_kwargs}")
            return model

        except ImportError as e:
            raise ImportError(
                f"Failed to import gradient model '{model_name}'. "
                f"Please ensure the 'gcn' package is installed and accessible. "
                f"Original error: {e}"
            )

    def _load_weights(self, weights_path: str):
        """Load pre-trained weights from file."""
        weights_path = Path(weights_path).expanduser().resolve()

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load state dict with strict=False to handle partial loading
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Warning: Missing keys when loading {self.model_name}:")
                print(f"  {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading {self.model_name}:")
                print(f"  {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

            print(f"Successfully loaded pre-trained weights from {weights_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {weights_path}: {e}")

    def _get_in_channels(self) -> int:
        """Infer input channels from the model's first layer."""
        try:
            # Try to find the first conv layer
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    return module.in_channels
            return 1  # Default fallback
        except:
            return 1  # Default fallback

    def _get_out_channels(self) -> int:
        """
        Get the adapter's output channels.
        The adapter always outputs 3 channels: [source, dx, dy]
        where the gradient model provides 2 channels (dx, dy) and we concatenate
        with the source image to produce 3 channels.
        """
        return 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - channel interpolation with optional resolution matching.

        Args:
            x: Input tensor of shape (B, in_channels, H, W) in range [0, 1]

        Returns:
            Output tensor of shape (B, 3, H, W)
            If target_resolution is set, output will be (B, 3, target_resolution, target_resolution)

        Output format: [original_image, dx, dy]
        - Channel 0: Original grayscale image [0, 1]
        - Channel 1: Horizontal gradient dx (scaled to [0, 1] if apply_scaling=True, else [-1, 1])
        - Channel 2: Vertical gradient dy (scaled to [0, 1] if apply_scaling=True, else [-1, 1])
        """
        # Store original input
        original = x  # [B, 1, H, W] in [0, 1]

        # Determine target size for output
        if self.target_resolution is not None:
            target_size = (self.target_resolution, self.target_resolution)
        else:
            target_size = original.shape[-2:]

        # Resize original to target size if needed
        if original.shape[-2:] != target_size:
            original = F.interpolate(
                original,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Forward through pre-trained gradient model
        gradients = self.model(x)  # [B, 2, H, W] with [dx, dy] in [-1, 1]

        # Ensure we have 2 gradient channels
        if gradients.shape[1] != 2:
            raise ValueError(
                f"Gradient model should output 2 channels [dx, dy], "
                f"got {gradients.shape[1]} channels"
            )

        # Resize gradients to match target size if needed
        # (gradient models may not preserve spatial dimensions due to striding)
        if gradients.shape[-2:] != target_size:
            gradients = F.interpolate(
                gradients,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Extract gradient channels
        dx, dy = gradients[:, 0:1], gradients[:, 1:2]

        # Apply linear scaling if enabled
        if self.apply_scaling:
            # Linear scaling: [-1, 1] → [0, 1]
            # This preserves gradient magnitude and direction while making them
            # compatible with DinoV2 normalization (which will be applied later)
            dx = (dx + 1.0) / 2.0
            dy = (dy + 1.0) / 2.0

        # Concatenate: [original, dx, dy]
        x = torch.cat([original, dx, dy], dim=1)  # [B, 3, H, W]

        return x

    def freeze(self):
        """Freeze all parameters (useful for transfer learning)."""
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"Froze all parameters in {self.model_name}")

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        print(f"Unfroze all parameters in {self.model_name}")
