"""Training-aware RF-DETR wrappers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bacdetr.detr import (
    RFDETR,
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegPreview,
    RFDETRSmall,
)
from bacdetr_train.util.coco_classes import COCO_CLASSES
from bacdetr_train.util.early_stopping import EarlyStoppingCallback

from bacdetr_train.config import SegmentationTrainConfig, TrainConfig
from bacdetr_train.trainer import Model
from bacdetr_train.util.metrics import (
    MetricsPlotSink,
    MetricsTensorBoardSink,
    MetricsWandBSink,
)


class TrainableRFDETR(RFDETR):
    """Mixin that reintroduces training support for RF-DETR models."""

    train_config_cls = TrainConfig

    def __init__(self, **kwargs: Any) -> None:
        self.callbacks = defaultdict(list)
        super().__init__(**kwargs)

    def get_model(self, config):
        return Model(**config.model_dump())

    def get_train_config(self, **kwargs):
        return self.train_config_cls(**kwargs)

    def train(self, **kwargs):
        config = self.get_train_config(**kwargs)
        return self.train_from_config(config, **kwargs)

    def train_from_config(self, config, **kwargs):
        if config.dataset_file == "coco":
            class_names = COCO_CLASSES
            num_classes = 90
            self.model.class_names = class_names
        else:
            raise ValueError(f"Invalid dataset file: {config.dataset_file}")

        if self.model_config.num_classes != num_classes:
            self.model.reinitialize_detection_head(num_classes)

        train_config = config.model_dump()
        model_config = self.model_config.model_dump()
        model_config.pop("num_classes", None)
        model_config.pop("class_names", None)

        if train_config.get("class_names") is None:
            train_config["class_names"] = class_names

        for k in list(train_config.keys()):
            if k in model_config:
                model_config.pop(k)

        for k in list(kwargs.keys()):
            if k in model_config:
                model_config.pop(k)

        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}

        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
            self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

        if config.wandb:
            metrics_wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir,
                project=config.project,
                run=config.run,
                config=config.model_dump(),
            )
            self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
            self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

        if config.early_stopping:

            early_stopping_callback = EarlyStoppingCallback(
                model=self.model,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema,
                segmentation_head=config.segmentation_head,
            )
            self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

        self.model.train(
            **all_kwargs,
            callbacks=self.callbacks,
        )


class TrainableRFDETRBase(TrainableRFDETR, RFDETRBase):
    pass


class TrainableRFDETRLarge(TrainableRFDETR, RFDETRLarge):
    pass


class TrainableRFDETRMedium(TrainableRFDETR, RFDETRMedium):
    pass


class TrainableRFDETRSmall(TrainableRFDETR, RFDETRSmall):
    pass


class TrainableRFDETRNano(TrainableRFDETR, RFDETRNano):
    pass


class TrainableRFDETRSegPreview(TrainableRFDETR, RFDETRSegPreview):
    train_config_cls = SegmentationTrainConfig
