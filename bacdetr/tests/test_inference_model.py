"""
Unit tests for bacdetr.model.InferenceModel.

Focus areas:
- _build_inference_args: default merging and override behaviour
- InferenceModel.__init__: attribute wiring
- _load_weights: checkpoint handling (matching / mismatching num_classes, class_names,
  corrupt file)
- BacDETR.get_model: no longer imports bacdetr_train
"""

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config_dict(**overrides):
    """Return a config dict that satisfies all required ModelConfig fields."""
    base = dict(
        encoder="dinov2_windowed_small",
        out_feature_indexes=[2, 5, 8, 11],
        dec_layers=3,
        projector_scale=["P4"],
        hidden_dim=256,
        patch_size=14,
        num_windows=4,
        sa_nheads=8,
        ca_nheads=16,
        dec_n_points=2,
        resolution=560,
        positional_encoding_size=37,
        # ModelConfig fields with non-trivial defaults
        num_classes=3,
        device="cpu",
        pretrain_weights=None,
        group_detr=13,
        gradient_checkpointing=False,
        two_stage=True,
        bbox_reparam=True,
        lite_refpoint_refine=True,
        layer_norm=True,
        amp=True,
        ia_bce_loss=True,
        cls_loss_coef=1.0,
        segmentation_head=False,
        mask_downsample_ratio=4,
        in_chans=3,
        channel_adapter=None,
        normalize_after_adapter=True,
        normalization_stats="dinov2",
        num_queries=300,
        num_select=300,
    )
    base.update(overrides)
    return base


def _make_checkpoint(num_classes_bias: int, with_class_names=False):
    """Build a minimal fake checkpoint dict."""
    checkpoint = {
        "model": {
            "class_embed.bias": torch.zeros(num_classes_bias),
            "class_embed.weight": torch.zeros(num_classes_bias, 256),
        }
    }
    if with_class_names:
        fake_args = SimpleNamespace(class_names=["rod", "coccus", "spore"])
        checkpoint["args"] = fake_args
    return checkpoint


# ---------------------------------------------------------------------------
# _build_inference_args
# ---------------------------------------------------------------------------

class TestBuildInferenceArgs:
    def test_defaults_are_applied(self):
        from bacdetr.model import _build_inference_args, _INFERENCE_DEFAULTS

        cfg = _minimal_config_dict()
        ns = _build_inference_args(cfg)

        for key, val in _INFERENCE_DEFAULTS.items():
            if key not in cfg:
                assert getattr(ns, key) == val, f"Default for {key!r} not applied"

    def test_config_overrides_defaults(self):
        from bacdetr.model import _build_inference_args

        cfg = _minimal_config_dict(num_queries=42, num_select=42, dropout=0.5)
        ns = _build_inference_args(cfg)

        assert ns.num_queries == 42
        assert ns.num_select == 42
        assert ns.dropout == 0.5

    def test_returns_namespace(self):
        from bacdetr.model import _build_inference_args

        ns = _build_inference_args(_minimal_config_dict())
        assert isinstance(ns, argparse.Namespace)

    def test_config_fields_present(self):
        from bacdetr.model import _build_inference_args

        cfg = _minimal_config_dict()
        ns = _build_inference_args(cfg)

        for key in cfg:
            assert hasattr(ns, key), f"Config field {key!r} missing from namespace"


# ---------------------------------------------------------------------------
# InferenceModel.__init__ — mocked build_model / PostProcess
# ---------------------------------------------------------------------------

class TestInferenceModelInit:
    """Verify __init__ wiring without building the real model."""

    def _make_model(self, **kwargs):
        """Instantiate InferenceModel with build_model and PostProcess mocked."""
        fake_nn_model = MagicMock()  # no spec: LWDETR has custom methods not on nn.Module
        fake_nn_model.to.return_value = fake_nn_model

        cfg = _minimal_config_dict(**kwargs)

        with (
            patch("bacdetr.model.build_model", return_value=fake_nn_model) as mock_bm,
            patch("bacdetr.model.PostProcess") as mock_pp,
        ):
            from bacdetr.model import InferenceModel
            m = InferenceModel(**cfg)
            return m, mock_bm, mock_pp

    def test_resolution_set(self):
        m, _, _ = self._make_model(resolution=448)
        assert m.resolution == 448

    def test_device_set(self):
        m, _, _ = self._make_model(device="cpu")
        assert m.device == torch.device("cpu")

    def test_stop_early_false(self):
        m, _, _ = self._make_model()
        assert m.stop_early is False

    def test_postprocess_created_with_num_select(self):
        _m, _bm, mock_pp = self._make_model(num_select=200)
        mock_pp.assert_called_once_with(num_select=200)

    def test_build_model_called_with_namespace(self):
        _m, mock_bm, _pp = self._make_model()
        assert mock_bm.call_count == 1
        passed_args = mock_bm.call_args[0][0]
        assert isinstance(passed_args, argparse.Namespace)

    def test_no_load_weights_when_pretrain_is_none(self):
        with (
            patch("bacdetr.model.build_model", return_value=MagicMock(spec=nn.Module)),
            patch("bacdetr.model.PostProcess"),
            patch("bacdetr.model.InferenceModel._load_weights") as mock_lw,
        ):
            from bacdetr.model import InferenceModel
            InferenceModel(**_minimal_config_dict(pretrain_weights=None))
            mock_lw.assert_not_called()

    def test_load_weights_called_when_pretrain_given(self):
        fake_nn_model = MagicMock(spec=nn.Module)
        fake_nn_model.to.return_value = fake_nn_model

        with (
            patch("bacdetr.model.build_model", return_value=fake_nn_model),
            patch("bacdetr.model.PostProcess"),
            patch("bacdetr.model.InferenceModel._load_weights") as mock_lw,
        ):
            from bacdetr.model import InferenceModel
            InferenceModel(**_minimal_config_dict(pretrain_weights="weights.pth"))
            mock_lw.assert_called_once()


# ---------------------------------------------------------------------------
# InferenceModel._load_weights
# ---------------------------------------------------------------------------

class TestLoadWeights:
    def _make_inference_model_stub(self, num_classes=3):
        """Return an InferenceModel with a real-ish inner model, without building the real backbone."""
        fake_nn_model = MagicMock()  # no spec: custom method not on nn.Module
        fake_nn_model.to.return_value = fake_nn_model

        cfg = _minimal_config_dict(num_classes=num_classes)

        with (
            patch("bacdetr.model.build_model", return_value=fake_nn_model),
            patch("bacdetr.model.PostProcess"),
        ):
            from bacdetr.model import InferenceModel
            m = InferenceModel(**cfg)
        return m

    def test_matching_num_classes_loads_state_dict(self):
        m = self._make_inference_model_stub(num_classes=3)
        # checkpoint has 4 classes (3+1 = 4)
        checkpoint = _make_checkpoint(num_classes_bias=4)

        with patch("torch.load", return_value=checkpoint):
            args = argparse.Namespace(
                pretrain_weights="fake.pth",
                num_classes=3,
            )
            m._load_weights(args)

        m.model.load_state_dict.assert_called_once_with(checkpoint["model"], strict=False)
        # reinitialize should NOT have been called
        m.model.reinitialize_detection_head.assert_not_called()

    def test_mismatching_num_classes_reinitializes_head(self):
        m = self._make_inference_model_stub(num_classes=3)
        # checkpoint has 91 classes (COCO), model expects 4
        checkpoint = _make_checkpoint(num_classes_bias=91)

        with patch("torch.load", return_value=checkpoint):
            args = argparse.Namespace(
                pretrain_weights="fake.pth",
                num_classes=3,
            )
            m._load_weights(args)

        m.model.reinitialize_detection_head.assert_called_once_with(91)
        m.model.load_state_dict.assert_called_once()

    def test_class_names_extracted_from_checkpoint(self):
        m = self._make_inference_model_stub(num_classes=3)
        checkpoint = _make_checkpoint(num_classes_bias=4, with_class_names=True)

        with patch("torch.load", return_value=checkpoint):
            args = argparse.Namespace(
                pretrain_weights="fake.pth",
                num_classes=3,
            )
            m._load_weights(args)

        assert m.class_names == ["rod", "coccus", "spore"]
        assert m.args.class_names == ["rod", "coccus", "spore"]

    def test_corrupt_file_raises_runtime_error(self):
        m = self._make_inference_model_stub(num_classes=3)

        with patch("torch.load", side_effect=RuntimeError("file corrupted")):
            args = argparse.Namespace(
                pretrain_weights="bad.pth",
                num_classes=3,
            )
            with pytest.raises(RuntimeError, match="Failed to load pretrain weights from 'bad.pth'"):
                m._load_weights(args)

    def test_missing_file_raises_runtime_error(self):
        m = self._make_inference_model_stub(num_classes=3)

        with patch("torch.load", side_effect=FileNotFoundError("no such file")):
            args = argparse.Namespace(
                pretrain_weights="missing.pth",
                num_classes=3,
            )
            with pytest.raises(RuntimeError, match="Failed to load pretrain weights"):
                m._load_weights(args)


# ---------------------------------------------------------------------------
# BacDETR.get_model — no bacdetr_train import required
# ---------------------------------------------------------------------------

class TestBacDETRGetModel:
    """Verify the base get_model() now returns InferenceModel without bacdetr_train."""

    @staticmethod
    def _make_concrete_bac():
        """Return a concrete BacDETR subclass instance, bypassing __init__."""
        from bacdetr.detr import BacDETRBase

        # BacDETRBase still has abstract train / train_from_config; create a
        # minimal concrete subclass so object.__new__ works.
        class _Concrete(BacDETRBase):
            def train(self, **kwargs): pass
            def train_from_config(self, *args, **kwargs): pass

        return object.__new__(_Concrete)

    def test_get_model_returns_inference_model(self):
        """get_model() must return an InferenceModel, not a trainer.Model."""
        from bacdetr.model import InferenceModel
        from bacdetr.config import BacDETRBaseConfig

        fake_nn_model = MagicMock()
        fake_nn_model.to.return_value = fake_nn_model

        with (
            patch("bacdetr.model.build_model", return_value=fake_nn_model),
            patch("bacdetr.model.PostProcess"),
        ):
            config = BacDETRBaseConfig(num_classes=3, pretrain_weights=None)
            bac = self._make_concrete_bac()
            result = bac.get_model(config)

        assert isinstance(result, InferenceModel)

    def test_get_model_does_not_import_bacdetr_train(self):
        """get_model() must not trigger a bacdetr_train import."""
        from bacdetr.config import BacDETRBaseConfig

        fake_nn_model = MagicMock()
        fake_nn_model.to.return_value = fake_nn_model

        # Poison sys.modules so any accidental bacdetr_train import raises ImportError
        bacdetr_train_backup = {
            k: v for k, v in sys.modules.items() if k.startswith("bacdetr_train")
        }
        for k in bacdetr_train_backup:
            sys.modules[k] = None  # type: ignore[assignment]

        try:
            with (
                patch("bacdetr.model.build_model", return_value=fake_nn_model),
                patch("bacdetr.model.PostProcess"),
            ):
                config = BacDETRBaseConfig(num_classes=3, pretrain_weights=None)
                bac = self._make_concrete_bac()
                # Must not raise ImportError for bacdetr_train
                bac.get_model(config)
        finally:
            for k in bacdetr_train_backup:
                sys.modules[k] = bacdetr_train_backup[k]
            for k in list(sys.modules):
                if k.startswith("bacdetr_train") and k not in bacdetr_train_backup:
                    del sys.modules[k]


# ---------------------------------------------------------------------------
# reinitialize_detection_head delegation
# ---------------------------------------------------------------------------

class TestReinitializeDetectionHead:
    def test_delegates_to_inner_model(self):
        from bacdetr.model import InferenceModel

        fake_nn_model = MagicMock()  # no spec: custom method not on nn.Module
        fake_nn_model.to.return_value = fake_nn_model

        with (
            patch("bacdetr.model.build_model", return_value=fake_nn_model),
            patch("bacdetr.model.PostProcess"),
        ):
            m = InferenceModel(**_minimal_config_dict())

        m.reinitialize_detection_head(5)
        m.model.reinitialize_detection_head.assert_called_once_with(5)
