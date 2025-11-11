# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from RF-DETR (https://github.com/roboflow/rf-detr)
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# ------------------------------------------------------------------------


import os
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Export config classes (core inference package)
from bacdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRSegPreviewConfig,
)

# TODO: Refactor detr.py to separate inference from training logic
# from bacdetr.detr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRSegPreview

__all__ = [
    "ModelConfig",
    "RFDETRBaseConfig",
    "RFDETRLargeConfig",
    "RFDETRNanoConfig",
    "RFDETRSmallConfig",
    "RFDETRMediumConfig",
    "RFDETRSegPreviewConfig",
]
