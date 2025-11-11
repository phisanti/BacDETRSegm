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
import torch
import os

# Roboflow dependencies (to be made optional or removed)
try:
    from rf100vl import get_rf100vl_projects
    import roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Warning: Roboflow dependencies not available. CLI functionality limited.")

def download_dataset(rf_project: roboflow.Project, dataset_version: int):
    versions = rf_project.versions()
    if dataset_version is not None:
        versions = [v for v in versions if v.version == str(dataset_version)]
        if len(versions) == 0:
            raise ValueError(f"Dataset version {dataset_version} not found")
        version = versions[0]
    else:
        version = max(versions, key=lambda v: v.id)
    location = os.path.join("datasets/", rf_project.name + "_v" + version.version)
    if not os.path.exists(location):
        location = version.download(
            model_format="coco", location=location, overwrite=False
        ).location
    
    return location


def train_from_rf_project(rf_project, dataset_version: int):
    """
    Train from Roboflow project.
    TODO: Refactor to use bacdetr-train API instead of old rfdetr.RFDETRBase
    """
    if not ROBOFLOW_AVAILABLE:
        raise RuntimeError("Roboflow dependencies not available")

    location = download_dataset(rf_project, dataset_version)
    print(location)

    # TODO: Replace with bacdetr-train API
    raise NotImplementedError(
        "This CLI needs refactoring to work with the split bacdetr/bacdetr-train packages. "
        "Use bacdetr_train.trainer.Model directly for now."
    )


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
    parser.add_argument("--coco_dir", type=str, required=False)
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument("--workspace", type=str, required=False, default=None)
    parser.add_argument("--project_name", type=str, required=False, default=None)
    parser.add_argument("--dataset_version", type=int, required=False, default=None)
    args = parser.parse_args()
    
    if args.coco_dir is not None:
        train_from_coco_dir(args.coco_dir)
        return

    if (args.workspace is None and args.project_name is not None) or (
        args.workspace is not None and args.project_name is None
    ):
        raise ValueError(
            "Either both workspace and project_name must be provided or none of them"
        )

    if args.workspace is not None:
        rf = roboflow.Roboflow(api_key=args.api_key)
        project = rf.workspace(args.workspace).project(args.project_name)
    else:
        projects = get_rf100vl_projects(api_key=args.api_key)
        project = projects[0].rf_project

    train_from_rf_project(project, args.dataset_version)


if __name__ == "__main__":
    trainer()
