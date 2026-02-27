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
    ChannelAdapterConfig,
    BacDETRBaseConfig,
    BacDETRLargeConfig,
    BacDETRNanoConfig,
    BacDETRSmallConfig,
    BacDETRMediumConfig,
    BacDETRSegConfig,
    BacDETRRecallerConfig,
    GradBacDETRConfig,
)

# Export model classes
from bacdetr.detr import (
    BacDETR,
    BacDETRBase,
    BacDETRLarge,
    BacDETRNano,
    BacDETRSmall,
    BacDETRMedium,
    BacDETRSeg,
    GradBacDETR,
    BacDETRRecaller,
)

__all__ = [
    # Base classes
    "ModelConfig",
    "ChannelAdapterConfig",
    "BacDETR",
    # Config classes
    "BacDETRBaseConfig",
    "BacDETRLargeConfig",
    "BacDETRNanoConfig",
    "BacDETRSmallConfig",
    "BacDETRMediumConfig",
    "BacDETRSegConfig",
    "BacDETRRecallerConfig",
    "GradBacDETRConfig",
    # Model classes
    "BacDETRBase",
    "BacDETRLarge",
    "BacDETRNano",
    "BacDETRSmall",
    "BacDETRMedium",
    "BacDETRSeg",
    "GradBacDETR",
    "BacDETRRecaller",
]
