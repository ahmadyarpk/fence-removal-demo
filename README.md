---
title: Fence Removal Demo
emoji: 🚧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Deep Learning-Based Fence Removal

A two-stage deep learning pipeline that automatically detects and removes fences from photographs.

## Pipeline

```
Input Image
     │
     ▼
┌─────────────────────┐
│  FenceDetector       │  ← DeepLabV3-ResNet101 (fine-tuned)
│  (Segmentation)      │     Outputs per-pixel fence probability mask
└─────────────────────┘
     │  mask
     ▼
┌─────────────────────┐
│  LaMaInpaintNet      │  ← LaMa-inspired encoder-decoder
│  (Inpainting)        │     Fills masked regions with plausible content
└─────────────────────┘
     │
     ▼
  Clean Image
```

## Architecture

| Component | Details |
|---|---|
| **Fence Detector** | DeepLabV3-ResNet101, classifier head replaced with 1-channel conv |
| **Inpaint Net** | Encoder → bottleneck → decoder (ConvTranspose upsampling) |
| **Input** | 256 × 256 RGB |
| **Training** | 150 epochs, BCE+Dice (mask) + L1 + Perceptual + GAN (inpainting) |

## Running Locally

```bash
git clone https://github.com/your-username/fence-removal-demo
cd fence-removal-demo
pip install -r requirements.txt
python app.py
```

> Set `HF_REPO_ID` in `app.py` to point to your Hugging Face model repo.

## Dataset

Tested on challenging real-world fenced images including zoo, sports, and wildlife scenarios.
