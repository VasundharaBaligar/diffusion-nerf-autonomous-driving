# 🚗 Diffusion-NeRF Scene Reconstruction for Autonomous Driving

A full end-to-end pipeline for **clean 3D scene reconstruction** from driving footage by combining instance segmentation, neural radiance fields, and diffusion-based inpainting.

![Pipeline](https://img.shields.io/badge/Pipeline-MMCV%20%7C%20NeRF%20%7C%20Diffusion-blue)
![Dataset](https://img.shields.io/badge/Dataset-KITTI-green)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)

---
### Before/After Comparison (Original → Mask → Inpainted)
comparison_v2.mp4
### NeRF Novel View Synthesis
render_hq.mp4
---
## 🎯 Motivation

Neural Radiance Fields (NeRF) assume a **static world** — but real driving scenes contain dynamic objects (cars, pedestrians) that move between frames, causing ghosting and blurring in 3D reconstruction.

This pipeline solves that by:
1. **Detecting** dynamic objects with MMCV-based segmentation
2. **Removing** them from every frame
3. **Inpainting** the holes with Stable Diffusion
4. **Reconstructing** a clean, artifact-free 3D scene with NeRF

---

## 🏗️ Pipeline Architecture

```
KITTI Driving Frames
        ↓
┌─────────────────────┐
│  MMCV + Mask RCNN   │  → Detects & segments dynamic objects (cars, people)
└─────────────────────┘
        ↓
┌─────────────────────┐
│    Binary Masking   │  → Creates per-frame masks of dynamic regions
└─────────────────────┘
        ↓
┌─────────────────────────────────┐
│  Stable Diffusion Inpainting    │  → Fills masked regions with realistic background
└─────────────────────────────────┘
        ↓
┌─────────────────────┐
│   Instant-NGP NeRF  │  → Trains 3D scene representation on clean frames
└─────────────────────┘
        ↓
  Novel View Synthesis of Clean Static Scene
```

---

## 📦 Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Image Processing | **MMCV 2.1.0** | Preprocessing, normalization |
| Object Detection | **Mask RCNN (torchvision)** | Instance segmentation |
| Diffusion Inpainting | **Stable Diffusion (runwayml)** | Background completion |
| 3D Reconstruction | **Instant-NGP (nerfstudio)** | Novel view synthesis |
| Dataset | **KITTI Raw** | Driving sequences |
| Framework | **PyTorch 2.10** | Core ML framework |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install mmcv==2.1.0 mmdet mmengine
pip install diffusers transformers accelerate
pip install nerfstudio
pip install opencv-python Pillow
```

### 2. Download KITTI Data
```python
import wget
wget.download(
  "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"
)
```

### 3. Run Full Pipeline
```bash
# Step 1: Detection + Masking
python scripts/detect_and_mask.py --data ./data/kitti --output ./masks

# Step 2: Diffusion Inpainting
python scripts/inpaint.py --frames ./nerf_input --masks ./masks --output ./inpainted

# Step 3: NeRF Training
ns-process-data images --data ./inpainted --output-dir ./nerf_data
ns-train instant-ngp --data ./nerf_data --max-num-iterations 20000

# Step 4: Render Novel Views
ns-render interpolate --load-config ./nerf_output/**/config.yml \
  --output-path ./render.mp4 --interpolation-steps 120
```

---

## 📊 Results

### Detection & Masking
- **108 frames** processed from KITTI sequence
- **8.7 frames/second** on NVIDIA A100
- Detects: cars, trucks, buses, motorcycles, pedestrians

### NeRF Training
- **20,000 iterations** on NVIDIA A100
- Clean novel view synthesis with no dynamic object ghosting

### Diffusion Inpainting
- Model: `runwayml/stable-diffusion-inpainting`
- 50 inference steps per frame
- Guidance scale: 15.0

---


## 📚 References

- [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934)
- [Instant Neural Graphics Primitives](https://arxiv.org/abs/2201.05989)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [MMCV: OpenMMLab Computer Vision Foundation](https://github.com/open-mmlab/mmcv)
- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)

---

## 👤 Author

Built as a research project targeting foundation model applications in autonomous driving, directly addressing challenges in 3D scene understanding, diffusion models, and sensor fusion.
