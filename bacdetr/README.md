# BacDETR - Inference Package

Lightweight inference package for Bacterial Detection Transformer models.

## Installation

```bash
pip install .
```

For visualization support:
```bash
pip install .[viz]
```

## Usage

```python
from bacdetr import RFDETRBase
from bacdetr.config import RFDETRBaseConfig

# Create model config
config = RFDETRBaseConfig(
    num_classes=3,
    pretrain_weights="path/to/checkpoint.pth"
)

# Initialize model (TODO: refactor inference API)
# model = RFDETRBase(**config.dict())
```

## Note

This package is under active refactoring. The inference API is being separated from training logic.
