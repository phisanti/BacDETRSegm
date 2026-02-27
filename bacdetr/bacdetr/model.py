# ------------------------------------------------------------------------
# BacDETR
# Copyright (c) 2025. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import argparse
import os
import torch

from bacdetr.models import build_model, PostProcess

_INFERENCE_DEFAULTS = dict(
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    drop_path=0.0,
    position_embedding="sine",
    use_cls_token=False,
    freeze_encoder=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    pretrained_encoder=None,
    dim_feedforward=2048,
    decoder_norm="LN",
    dropout=0.0,
    aux_loss=True,
    focal_alpha=0.25,
    freeze_batch_norm=False,
    num_queries=300,
    num_select=300,
    encoder_only=False,
    backbone_only=False,
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
)


def _build_inference_args(config_dict: dict) -> argparse.Namespace:
    args = dict(_INFERENCE_DEFAULTS)
    args.update(config_dict)  # ModelConfig fields take precedence
    return argparse.Namespace(**args)


class InferenceModel:
    """Build-and-load wrapper for pure inference without bacdetr_train."""

    def __init__(self, **kwargs):
        args = _build_inference_args(kwargs)
        self.args = args
        self.resolution = args.resolution
        self.device = torch.device(args.device)
        self.model = build_model(args)
        if args.pretrain_weights is not None:
            self._load_weights(args)
        self.model = self.model.to(self.device)
        self.postprocess = PostProcess(num_select=getattr(args, "num_select", 300))
        self.stop_early = False

    def _load_weights(self, args) -> None:
        try:
            checkpoint = torch.load(
                args.pretrain_weights, map_location="cpu", weights_only=False
            )
        except Exception as e:
            # maybe_download_pretrain_weights() was already called by BacDETR.__init__,
            # so re-raise rather than silently re-downloading.
            raise RuntimeError(
                f"Failed to load pretrain weights from '{args.pretrain_weights}': {e}"
            ) from e

        if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
            self.args.class_names = checkpoint["args"].class_names
            self.class_names = checkpoint["args"].class_names

        checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
        if checkpoint_num_classes != args.num_classes + 1:
            self.reinitialize_detection_head(checkpoint_num_classes)

        self.model.load_state_dict(checkpoint["model"], strict=False)

    def reinitialize_detection_head(self, num_classes: int) -> None:
        self.model.reinitialize_detection_head(num_classes)

    def export(
        self,
        output_dir="output",
        infer_dir=None,
        simplify=False,
        backbone_only=False,
        opset_version=17,
        verbose=True,
        force=False,
        shape=None,
        batch_size=1,
        **kwargs,
    ):
        """Export to ONNX.  onnxsim simplification requires onnxsim installed."""
        from copy import deepcopy
        from pathlib import Path

        import numpy as np
        import torchvision.transforms.functional as TF
        from PIL import Image

        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        elif shape[0] % 14 != 0 or shape[1] % 14 != 0:
            raise ValueError("Shape must be divisible by 14")

        # Build a dummy input (replaces make_infer_image from bacdetr_train.deploy.export)
        if infer_dir is None:
            dummy = np.random.randint(0, 256, (shape[0], shape[1], 3), dtype=np.uint8)
            image = Image.fromarray(dummy, mode="RGB")
        else:
            image = Image.open(infer_dir).convert("RGB")
        image = image.resize((shape[1], shape[0]))
        t = TF.to_tensor(image)
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        input_tensors = torch.stack([t] * batch_size).to(device)

        input_names = ["input"]
        output_names = ["features"] if backbone_only else ["dets", "labels"]

        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            elif self.args.segmentation_head:
                outputs = model(input_tensors)
                print(
                    f"Output shapes — boxes: {outputs['pred_boxes'].shape}, "
                    f"logits: {outputs['pred_logits'].shape}, "
                    f"masks: {outputs['pred_masks'].shape}"
                )
            else:
                outputs = model(input_tensors)
                print(
                    f"Output shapes — boxes: {outputs['pred_boxes'].shape}, "
                    f"logits: {outputs['pred_logits'].shape}"
                )

        model.cpu()
        input_tensors = input_tensors.cpu()

        # ONNX export (replaces export_onnx from bacdetr_train.deploy.export)
        export_name = "backbone_model" if backbone_only else "inference_model"
        output_file = str(output_dir / f"{export_name}.onnx")
        if hasattr(model, "export"):
            model.export()
        torch.onnx.export(
            model,
            input_tensors,
            output_file,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            verbose=verbose,
            opset_version=opset_version,
            dynamic_axes=None,
        )
        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            try:
                import onnx
                import onnxsim
            except ImportError as exc:
                raise ImportError(
                    "ONNX simplification requires onnxsim and onnx. "
                    "Install with: pip install onnxsim onnx"
                ) from exc
            sim_output_file = output_file.replace(".onnx", ".sim.onnx")
            model_opt, check_ok = onnxsim.simplify(output_file)
            if not check_ok:
                raise RuntimeError("Failed to simplify ONNX model.")
            onnx.save(model_opt, sim_output_file)
            print(f"Successfully simplified ONNX model to: {sim_output_file}")

        self.model = self.model.to(device)
        print("ONNX export completed successfully")
