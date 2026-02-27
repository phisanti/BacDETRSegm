# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
onnx optimizer and symbolic registry
"""
from . import optimizer as optimizer
from . import symbolic as symbolic

from .optimizer import OnnxOptimizer as OnnxOptimizer
from .symbolic import CustomOpSymbolicRegistry as CustomOpSymbolicRegistry