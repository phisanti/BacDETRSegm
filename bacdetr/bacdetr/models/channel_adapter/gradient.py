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
    to predict gradient flows from grayscale images. These gradient representations are
    used as pseudo-RGB input for the backbone.

    Args:
        model_name: Name of the pre-trained model (e.g., "gradconvnext_atto", "gradconvunext_femto")
        weights_path: Path to the pre-trained weights file
        target_resolution: Optional target resolution for upsampling (e.g., 312, 384, 432)
        freeze: Whether to freeze the model after loading (default: False)
    """
    def __init__(
        self,
        model_name: str,
        weights_path: Optional[str] = None,
        target_resolution: Optional[int] = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.weights_path = weights_path
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
        Infer output channels from the model's last layer.
        Gradient models typically output 2 channels (dx, dy) or 3 channels (dx, dy, magnitude).
        """
        try:
            # Try to find the last conv layer
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                out_ch = last_conv.out_channels
                # If gradient model outputs 2 channels, we need to adapt to 3 for RGB
                if out_ch == 2:
                    print(f"Gradient model outputs 2 channels (dx, dy), will replicate to 3 for RGB")
                    return 2  # Will be handled in forward()
                return out_ch
            return 3  # Default fallback
        except:
            return 3  # Default fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, 3, H', W')
            where H', W' = target_resolution if specified, else H, W
        """
        # Forward through pre-trained gradient model
        x = self.model(x)

        # If output is 2 channels (dx, dy), replicate to 3 channels for RGB compatibility
        if x.shape[1] == 2:
            # Option 1: Replicate the first channel [dx, dy, dx]
            # Option 2: Compute magnitude and use [dx, dy, magnitude]
            # We'll use option 2 for better representation
            magnitude = torch.sqrt(x[:, 0:1]**2 + x[:, 1:2]**2)
            x = torch.cat([x, magnitude], dim=1)

        # Ensure output is 3 channels
        if x.shape[1] > 3:
            # If model outputs more than 3 channels, take first 3
            x = x[:, :3]
        elif x.shape[1] < 3:
            # If less than 3, pad with zeros
            padding = torch.zeros(x.shape[0], 3 - x.shape[1], x.shape[2], x.shape[3],
                                 device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

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
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"Froze all parameters in {self.model_name}")

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        print(f"Unfroze all parameters in {self.model_name}")
