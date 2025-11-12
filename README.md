# BacDETRSegm: Bacterial Instance Segmentation with Detection Transformers

> Repository for the BacDETRSegm architecture, adapters, and training tooling, extending [RF-DETR](https://github.com/roboflow/rf-detr) to bright-field microscopy.

---

## 1. Project Aim

Provide reusable code, configs, and scripts to train and evaluate BacDETRSegm: a lightweight RF-DETR–based instance segmentation model targeting single-channel bright-field microscopy. To leverage pre-trained DINOv2 backbones on grayscale inputs, we introduce a family of adapters (ResUnit, ConvNeXt, GradientConvNeXt) that remap microscope intensities into feature-rich pseudo-RGB tensors.

**Channel adapters.** Each adapter acts as a learnable front-end that ingests any arbitrary _N_-channel scientific image stack—microscopy, fluorescence multiplexing, phase contrast, hyperspectral slices, etc.—and emits exactly three channels so the frozen DINOv2 weights, trained on 3-channel RGB natural images, can be reused without architectural changes.  
- `ResUnit Adapter`: shallow residual bottlenecks emphasize local contrast and quickly compress any number of input channels down to three, making it ideal when you want minimal extra latency.  
- `ConvNeXt Adapter`: ConvNeXt-style depthwise blocks provide a wider receptive field before the RGB hand-off, which helps when the scientific channels encode complementary cues (e.g., phase + bright-field).  
- `GradientConvNeXt Adapter`: extends the ConvNeXt variant with explicit gradient streams so the synthetic RGB output retains edge information critical for bacterial boundaries.

---

## 2. Problem Context

**Input**  
- Bright-field microscope crops (single channel).  
- Limited annotated corpus of bacterial colonies.  
- Strong morphology distortions (overlaps, division events, debris).

**Output**  
- Instance masks and boxes for single cells, dividing pairs, clumps, and debris.  
- Up to a few hundred predictions per crop depending on density.

---

## 3. Challenges & Strategy

| Challenge | Strategy |
|-----------|----------|
| Grayscale input vs. RGB-pretrained backbones | Adapter stack (ResUnit / ConvNeXt / GradientConvNeXt) augments intensity with learned spatial gradients |
| Large scale variance | Multi-scale projector and RF-DETR decoder preserve detail from small rods to larger clumps |
| Thin, elongated shapes | Segmentation head keeps fine-grained positional encoding and high-res mask logits |
| Overlaps and touching objects | Transformer queries with iterative refinement disentangle crowded regions |
| Small dataset | Freeze most of DINOv2 backbone, apply aggressive augmentation, keep model compact for regularization |

---

## 4. Architecture Overview

```
 Bright-field crop (1 channel)
        │
 Adapter (ResUnit | ConvNeXt | GradientConvNeXt)
  ↳ produces 3 enriched channels compatible with DINOv2 weights
        │
 DINOv2 backbone from RF-DETR (partially frozen)
        │
 Multi-scale projector
        │
 RF-DETR transformer decoder (two-stage)
        │
 Segmentation head from RF-DETR-Seg (Preview)
        │
 Boxes + masks + class scores
```

RF-DETR supplies the backbone, projector, transformer decoder, and segmentation head; this repo layers adapter modules, microscopy-specific configs, and training utilities on top.

---

## 5. Data & Training Pipeline

1. **Pre-processing** – Flat-field correction, contrast normalization, tiling into manageable crops.  
2. **Annotation** – Four instance classes (single, joint, clump, debris).  
3. **Augmentation** – Rotations, elastic warps, intensity jitter, mixup between sparse/dense crops.  
4. **Training** – Uses RF-DETR training stack with adapter-aware configs, partial weight freezing, cosine LR schedules, and mixed precision.  
5. **Evaluation** – Same scripts generate validation curves and qualitative overlays; numerical metrics will follow once experiments conclude.

---

## 6. Usage

### Environment
```bash
pip install -e .
```

### Dataset Preparation
```bash
python tools/prepare_dataset.py \
  --raw data/raw_brightfield \
  --labels data/annotations.json \
  --out data/bacteria_crops
```

### Train
```bash
python tools/train.py \
  --config configs/bacteria/rfdetr_nano_gradconv.yaml \
  --data data/bacteria_crops \
  --output runs/bacdetrsegm
```

### Evaluate / Infer
```bash
python tools/eval.py --checkpoint runs/bacdetrsegm/ckpt_latest.pth --split val

python tools/predict.py \
  --checkpoint runs/bacdetrsegm/ckpt_latest.pth \
  --image samples/crop.png \
  --save outputs/crop_vis.png
```

---

## 7. Attribution

BacDETRSegm reuses and builds upon the architecture, code, and training practices introduced in **RF-DETR** (Roboflow, Apache 2.0). Please cite RF-DETR alongside any publications or deployments using this repository. Inspiration also comes from LW-DETR, DINOv2, and Deformable DETR.
