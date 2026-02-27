# ------------------------------------------------------------------------
# BacDETR-Train CLI
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from RF-DETR (https://github.com/roboflow/rf-detr)
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

# TODO: This CLI is Roboflow-specific and needs refactoring for BacDETR
# The high-level training API needs to be redesigned to work with the split packages

import argparse



def train_from_coco_dir(coco_dir: str):
    """
    Train from COCO directory.
    TODO: Refactor to use bacdetr-train API instead of old rfdetr.RFDETRBase
    """
    # TODO: Replace with bacdetr-train API
    raise NotImplementedError(
        "This CLI needs refactoring to work with the split bacdetr/bacdetr-train packages. "
        "Use bacdetr_train.trainer.Model directly for now."
    )


def trainer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", type=str, required=True)
    args = parser.parse_args()

    train_from_coco_dir(args.coco_dir)


if __name__ == "__main__":
    trainer()
