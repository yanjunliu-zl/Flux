# 🎬 Seedance 2.0

**Dual-Branch Diffusion Transformer for Native Audio-Video Joint Generation**

An open-source recreation of ByteDance's Seedance 2.0 architecture for text-to-video-audio (T2VA) and image-to-video-audio (I2VA) generation.

> ⚠️ **Research Preview**: This is an open-source reference implementation based on the published architecture descriptions (Seedance 1.0/1.5/2.0 papers). It is NOT the official ByteDance implementation.

## Architecture

```
Text (T5 Encoder)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
  ┌──────────────┐                    ┌──────────────┐
  │ Vision Branch│◄── Cross-Modal ───►│ Audio Branch │
  │ (STDiT)      │     Bridge (CBGA)  │ (DiT)        │
  │ spatial+temporal                  │ self+text attn│
  │ attn + MM-RoPE│                   │  + 1D RoPE   │
  └──────┬───────┘                    └──────┬───────┘
         │                                   │
    VideoVAE                             AudioVAE
    (3D, 8×8×4)                       (Mel-spec, 2D)
         │                                   │
      Video                              Audio Waveform
```

### Key Components

| Component | Description |
|-----------|-------------|
| **VideoVAE** | 3D CausalConv3D autoencoder: 8× spatial + 4× temporal compression |
| **AudioVAE** | 2D Conv mel-spectrogram autoencoder |
| **DB-DiT** | Dual-Branch Diffusion Transformer with cross-modal bridge |
| **MM-RoPE** | Multi-Modal Rotary Position Embedding (3D video + 1D audio) |
| **CBGA** | Cross-Branch Gated Attention for frame-level AV synchronization |
| **Flow Matching** | Velocity field prediction with logit-normal timestep sampling |

### Model Variants

| Config | Layers | Hidden Dim | Heads | Params |
|--------|--------|-----------|-------|--------|
| **Small** | 12 | 768 | 12 | ~500M |
| **Base** | 24 | 1024 | 16 | ~1.3B |

## Quick Start

### Installation

```bash
pip install -e .
# Optional: flash attention for faster training
pip install flash-attn --no-build-isolation
```

### Data Preparation

```bash
# 1. Download videos
python -m seedance.tools.video_download --input urls.txt --output data/raw_videos/

# 2. Scene detection
python -m seedance.tools.scene_detection --input data/raw_videos/ --output data/clips/

# 3. Quality filtering
python -m seedance.tools.quality_filter --input data/clips/ --output data/filtered/

# 4. Generate captions
python -m seedance.tools.video_caption --input data/filtered/ --output data/captioned/

# 5. Extract audio
python -m seedance.tools.audio_extract --input data/filtered/ --output data/audio/

# 6. Build manifest
python -m seedance.tools.build_manifest \
    --video_dir data/filtered/ \
    --audio_dir data/audio/ \
    --captions data/captioned/captions.json \
    --output data/manifests/train.csv
```

### Training

```bash
# Stage 1: Video-only pretraining
python scripts/train.py --config configs/train/stage1_video_pretrain.yaml

# Stage 2: Audio-only pretraining
python scripts/train.py --config configs/train/stage2_audio_pretrain.yaml

# Stage 3: Audio-Video joint training
python scripts/train.py --config configs/train/stage3_av_joint.yaml

# Stage 4: High-resolution fine-tuning
python scripts/train.py --config configs/train/stage4_hires_finetune.yaml
```

### Inference

```bash
# Text-to-Video-Audio
python scripts/inference_t2va.py \
    --config configs/inference/t2va.yaml \
    --prompt "A dog running through a grassy field" \
    --output outputs/dog_field.mp4

# Image-to-Video-Audio
python scripts/inference_i2va.py \
    --config configs/inference/i2va.yaml \
    --image inputs/photo.jpg \
    --prompt "The person turns and smiles" \
    --output outputs/animated.mp4

# Gradio web demo
python scripts/gradio_app.py --checkpoint checkpoints/model.pt --port 7860
```

## Training Stages

| Stage | Description | Resolution | Steps |
|-------|------------|-----------|-------|
| 1 | Video pretraining | 256×256, 16-32fr | 500K |
| 2 | Audio pretraining | 16kHz mel-spec | 200K |
| 3 | AV joint training | 256×256, 16-32fr | 300K |
| 4 | Hi-res fine-tuning | 512×512, 64fr | 100K |

## Project Structure

```
seedance/
├── configs/          # YAML configuration files
├── scripts/          # Entry point scripts
├── seedance/
│   ├── models/       # VideoVAE, AudioVAE, DB-DiT, T5
│   ├── diffusion/    # Flow matching, schedulers, CFG
│   ├── data/         # Datasets, transforms, sampler
│   ├── pipelines/    # T2VA, I2VA inference
│   ├── training/     # Trainer, optimizer, FSDP
│   ├── loss/         # VAE loss, flow loss, sync loss
│   ├── utils/        # Config, checkpoint, video/audio I/O
│   └── tools/        # Data prep CLI tools
└── tests/            # Unit and integration tests
```

## Requirements

- Python 3.10+
- PyTorch 2.4+
- CUDA-capable GPU (24GB+ VRAM recommended)

## References

- [Seedance 2.0: Advancing Video Generation for World Complexity](https://arxiv.org/abs/2604.14148) — ByteDance Seed Team, 2026
- [Seedance 1.5 Pro: A Native Audio-Visual Joint Generation Foundation Model](https://arxiv.org/abs/2512.13507) — ByteDance Seed Team, 2025
- [Open-Sora: Democratizing Efficient Video Production](https://github.com/hpcaitech/Open-Sora) — HPC-AI Tech
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., 2023

## License

MIT
