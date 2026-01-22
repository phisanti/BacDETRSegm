# BacDETR-Train - Training Package

Training infrastructure for Bacterial Detection Transformer models.

## Installation

First install the inference package:
```bash
cd ../bacdetr
pip install .
```

Then install the training package:
```bash
cd ../bacdetr-train
pip install .
```

For all features (ONNX export, metrics logging):
```bash
pip install .[all]
```

## Usage

```python
from bacdetr_train.trainer import Model
from bacdetr.config import BacDETRBaseConfig
from bacdetr_train.config import TrainConfig

# Configure model
model_config = BacDETRBaseConfig(num_classes=3)

# Configure training
train_config = TrainConfig(
    dataset_dir="path/to/dataset",
    output_dir="output",
    epochs=50
)

# Train (API under refactoring)
# model = Model(**model_config.dict())
# model.train(**train_config.dict())
```

## Note

This package depends on `bacdetr` for model architecture. Both packages are under active refactoring.
