# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from RF-DETR (https://github.com/roboflow/rf-detr)
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# ------------------------------------------------------------------------


from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class ChannelAdapterConfig(BaseModel):
    """
    Configuration for channel adapter (grayscale -> pseudo-RGB conversion).

    Channel adapters are responsible ONLY for channel interpolation.
    Image resizing should be handled by the dataloader before the adapter.
    """
    enabled: bool = False
    adapter_type: Literal["residual", "convnext", "gradient", "pretrained"] = "residual"

    # For residual and convnext adapters
    in_channels: int = 1
    out_channels: int = 3
    num_blocks: int = 2
    intermediate_dim: int = 32
    drop_path: float = 0.0

    # For pretrained adapters (GradConvNeXt/UNeXt)
    model_name: Optional[str] = None  # e.g., "gradconvnext_atto", "gradconvunext_small"
    weights_path: Optional[str] = None
    target_resolution: Optional[int] = None

    # Training control
    freeze: bool = False


class ModelConfig(BaseModel):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    in_chans: int = 3  # Input channels (1 for grayscale, 3 for RGB)
    channel_adapter: Optional[ChannelAdapterConfig] = None

    # Post-adapter normalization control
    normalize_after_adapter: bool = True  # Apply normalization after adapter
    normalization_stats: Literal['dinov2', 'imagenet', 'none'] = 'dinov2'

    @model_validator(mode="before")
    def _sync_post_adapter_alias(cls, values):
        """Support legacy `post_adapter_normalization` configs."""
        legacy_key = 'post_adapter_normalization'
        if 'normalize_after_adapter' not in values and legacy_key in values:
            values['normalize_after_adapter'] = values[legacy_key]
        return values


class BacDETRBaseConfig(ModelConfig):
    """
    The configuration for a BacDETR Base model (scientific/grayscale images).
    """
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    resolution: int = 560
    positional_encoding_size: int = 37

class BacDETRLargeConfig(BacDETRBaseConfig):
    """
    The configuration for a BacDETR Large model.
    """
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_base"
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: Optional[str] = "rf-detr-large.pth"

class BacDETRNanoConfig(BacDETRBaseConfig):
    """
    The configuration for a BacDETR Nano model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"

class BacDETRSmallConfig(BacDETRBaseConfig):
    """
    The configuration for a BacDETR Small model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"

class BacDETRMediumConfig(BacDETRBaseConfig):
    """
    The configuration for a BacDETR Medium model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"

class BacDETRSegConfig(BacDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    num_classes: int = 90


class BacDETRRecallerConfig(BacDETRSegConfig):
    """
    Recall-optimized BacDETR segmentation config for small objects.

    Keeps the DINOv2-compatible backbone settings while increasing
    high-resolution feature usage and mask detail.
    """
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4"]
    num_select: int = 400
    mask_downsample_ratio: int = 2


class GradBacDETRConfig(BacDETRSegConfig):
    """
    BacDETR segmentation config with a fixed GradientConvNeXt adapter.
    """
    channel_adapter: ChannelAdapterConfig = Field(
        default_factory=lambda: ChannelAdapterConfig(
            enabled=True,
            adapter_type="gradient",
            model_name="gradconvnext_atto",
        )
    )

    @model_validator(mode="after")
    def _enforce_grad_adapter(self):
        adapter = self.channel_adapter or ChannelAdapterConfig(
            enabled=True,
            adapter_type="gradient",
            model_name="gradconvnext_atto",
        )
        adapter.enabled = True
        adapter.adapter_type = "gradient"
        adapter.model_name = "gradconvnext_atto"
        if adapter.target_resolution is None:
            adapter.target_resolution = self.resolution
        self.channel_adapter = adapter
        return self
