# 🎬 Flux

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

| Config | Layers | Hidden Dim | Heads | Params | Activated | Training VRAM |
|--------|--------|-----------|-------|--------|-----------|---------------|
| **Small** | 12 | 768 | 12 | ~0.4B | ~0.4B | ~25 GB |
| **Base** (default) | 24 | 1024 | 16 | ~1.6B | ~1.6B | ~72 GB |
| **30B Dense** | 48 | 2048 | 32 | ~30B | ~30B | ~160 GB (8×A100) |
| **30B MoE** | 48 | 2048 | 32 | ~30B | ~4B (13%) | ~80 GB (4×A100) |
| **200B MoE** | 48 | 4096 | 32 | ~200B | ~36B (18%) | NVMe offload (8×H100) |
| **4K 30s** | 32 | 4096 | 32 | ~200B | ~36B (18%) | NVMe + Seq Parallel |

> Config files in [configs/model/](configs/model/). Larger variants require distributed training via FSDP or DeepSpeed ZeRO-3. See [docs/ROADMAP_TO_SEEDANCE_2_5.md](docs/ROADMAP_TO_SEEDANCE_2_5.md) for scaling requirements.

## Quick Start

### Installation

```bash
# Install core dependencies (locked versions)
uv sync

# With development tools
uv sync --group dev

# With all optional dependencies
uv sync --group dev --extra flash-attn --extra tools --extra t5
```

### Data Preparation

#### VoxCeleb2 (Talking Faces)

VoxCeleb2 is the primary pretraining dataset, containing ~1.1M talking-face video clips
from 5,994 speakers. The official dataset requires registration at
[robots.ox.ac.uk/~vgg/data/voxceleb/](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

After downloading, you will have a **multi-part zip archive** (`vox2_dev_mp4_partaa` through `partai`,
~249 GB total) and a text annotations zip (`vox2_dev_txt.zip`, ~1.5 GB). These use `store`
compression (no actual compression), so you need ~250 GB free space for extraction.

```bash
# 1. Install 7-Zip for split archive handling (Windows)
winget install 7zip.7zip

# 2. Extract text annotations (~1.1M utterances → data/voxceleb2/txt/)
7z x data/voxceleb2/vox2_dev_txt.zip -odata/voxceleb2/ -y

# 3. Extract video split archives (~1.1M clips → data/voxceleb2/dev/mp4/)
#    7z auto-detects all 9 parts (partaa → partai)
7z x data/voxceleb2/dev_archives/vox2_dev_mp4_partaa -odata/voxceleb2/ -y
```

Extracted structure:
```
data/voxceleb2/
  dev/mp4/
    id00012/               ← speaker ID
      21Uxsk56VDQ/          ← YouTube video ID
        00001.mp4           ← individual clip (224×224, 25fps, ~3-15s)
        00002.mp4
        ...
  txt/
    id00012/
      21Uxsk56VDQ/
        00001.txt           ← text annotation per clip
        ...
```

After successful extraction, the original split archives can be safely deleted to
free ~250 GB:

```bash
rm -rf data/voxceleb2/dev_archives/
rm data/voxceleb2/vox2_dev_txt.zip
```

#### Manifest Generation

Generate a CSV manifest with full OpenCV metadata (frame count, resolution, FPS) for all
clips, plus SCRFD face detection on the first 500 samples:

```bash
python -m flux.tools.ingest_talking_data \
    --input_dir data/voxceleb2/dev/mp4/ \
    --dataset voxceleb \
    --output data/manifests/voxceleb_manifest.csv
# Output: 1,092,009 entries, 212 MB, ~2 hours
```

#### Merging All Data Sources

Stage 1 (video pretraining) benefits from data diversity. Merge all available sources —
VoxCeleb2, Pexels/WebVid, HDTF — into a single manifest:

```bash
python -c "
import csv, os

manifests = [
    'data/manifests/voxceleb_manifest.csv',
    'data/manifests/pexels_webvid_annotated.csv',
    'data/manifests/hdtf_manifest.csv',
]
columns = ['video_path','num_frames','height','width','fps','duration_s',
           'audio_path','caption_short','caption_long','caption_audio','speaker_id','dataset']

with open('data/manifests/train_stage1.csv','w',newline='',encoding='utf-8') as out:
    w = csv.DictWriter(out, fieldnames=columns, extrasaction='ignore')
    w.writeheader()
    for m in manifests:
        if os.path.exists(m):
            for row in csv.DictReader(open(m,'r',encoding='utf-8')):
                if os.path.exists(row.get('video_path','')):
                    w.writerow(row)
"
# Result: 1,093,167 videos, 209 MB
```

#### Data Summary

| Dataset | Videos | Size | Description |
|---------|--------|------|-------------|
| VoxCeleb2 | 1,092,009 | 254 GB | Talking faces, 5,994 speakers, 224×224 |
| WebVid/Pexels | 2,865 / 786 | ~5 GB | General web videos, stock footage |
| HDTF | 372 | 5.8 GB | High-res talking faces |
| **Total** | **1,093,167** | **~265 GB** | |

### Training

#### Launch Commands

```bash
# Single GPU
python -m flux.training --config configs/train/stage1_video_pretrain.yaml

# Single-node multi-GPU (auto-detect)
bash scripts/train.sh configs/train/stage1_video_pretrain.yaml 8

# Multi-node (2 nodes × 8 GPUs)
bash scripts/train.sh configs/train/stage1_video_pretrain.yaml 8 2 0 192.168.1.10   # node 0
bash scripts/train.sh configs/train/stage1_video_pretrain.yaml 8 2 1 192.168.1.10   # node 1

# Raw torchrun
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.10 --master_port=29500 \
    -m flux.training --config configs/train/stage1_video_pretrain.yaml
```

#### Stages

```bash
# Stage 1: Video-only pretraining
python -m flux.training --config configs/train/stage1_video_pretrain.yaml

# Stage 2: Audio-only pretraining
python -m flux.training --config configs/train/stage2_audio_pretrain.yaml

# Stage 3: Audio-Video joint training
python -m flux.training --config configs/train/stage3_av_joint.yaml

# Stage 4: High-resolution fine-tuning
python -m flux.training --config configs/train/stage4_hires_finetune.yaml
```

#### Distributed Architecture

| Feature | Implementation |
|---------|---------------|
| Strategy | FSDP (FULL_SHARD) with DDP fallback |
| Mixed precision | bf16 (AMP); fp16 via GradScaler |
| Data sharding | DistributedSampler (shuffle per-epoch) |
| Loss sync | `all_reduce` averaging across all ranks |
| Checkpoint | FSDP FULL_STATE_DICT consolidation (main process only) |
| Launcher | `torchrun` (PyTorch native) |
| Gradient checkpointing | FSDP activation checkpointing on `DualBranchBlock` |

#### Effective Batch Size

```
effective_batch = batch_size_per_gpu × num_gpus × gradient_accumulation_steps
                = 4 × 8 × 4 = 128 (typical 8-GPU setup)
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

### Physics Consistency (Cross-Cutting)

The model incorporates **6 complementary physics mechanisms** spanning training, inference, and monitoring:

| Mechanism | Phase | Paper | Description |
|-----------|-------|-------|-------------|
| **VPT** | Pre-training + Training | Zheng et al., 2026 | Role-aware captioning (`[agent: person] pushes [controlled: box]`) + modality-decoupled noise |
| **World Model Loss** | Training | VideoWorld 2 | Self-supervised future prediction + jerk minimization + collision detection |
| **PhysCorr (PhyDPO)** | Training | Wang et al., 2025 | 0.5B PhysicsRM reward model → DPO preference optimization |
| **PhysicsProbe** | Post-training | Esmati et al., 2026 | Linear probe decodes physical plausibility from DiT hidden states (81.27% accuracy) |
| **PhaseLock** | Inference | Han et al., ICML 2026 | 2-step coarse motion prior → Latent Delta Guidance lock (~1.06× overhead) |
| **CausalMotion** | Inference | Zhuang et al., 2026 | VLM-decomposed keyframes + object trajectories as soft constraints (training-free) |

See [docs/DESIGN.md](docs/DESIGN.md) §6 for full details.

## Project Structure

```
flux/
├── configs/              # YAML configuration files
│   ├── inference/        # T2VA, I2VA inference configs
│   ├── model/            # Model architecture configs (Small → 200B MoE → 4K 30s)
│   └── train/            # Training stage configs (Stage 1-4 + 30B + 200B variants)
├── scripts/              # Entry point scripts
├── flux/
│   ├── models/           # VideoVAE, AudioVAE, DB-DiT, T5, Face Analysis
│   │   ├── video_vae/    # 3D CausalConv3D autoencoder (8×8×4 compression)
│   │   ├── audio_vae/    # 2D Conv mel-spectrogram autoencoder
│   │   ├── db_dit/       # Dual-Branch DiT + CBGA + MM-RoPE + MoE + Sparse Attn
│   │   ├── text_encoder/ # T5 encoder (T5-XXL for 200B scale)
│   │   └── ...           # Face analysis, KP/LFA encoders, mouth ROI, reward model
│   ├── diffusion/        # Flow Matching, schedulers, CFG, noise schedules
│   ├── data/             # Datasets, transforms, bucket sampler, collation
│   │   └── annotation/   # Auto-labeling: scenario, motion quality, physics events
│   ├── pipelines/        # T2VA, I2VA, cascaded 4K 30s inference pipelines
│   ├── training/         # Trainer, SFT, RLHF/PPO, FSDP, DeepSpeed, optimizer
│   ├── physics/          # Physics consistency: PhaseLock, CausalMotion, PhysCorr, VPT, PhysicsProbe
│   ├── loss/             # Flow loss, VAE loss, sync loss, lip-sync loss, world model loss
│   ├── utils/            # Config, checkpoint, video/audio I/O
│   └── tools/            # Data downloaders, manifest builders, quality filters, captioners
├── tests/                # Unit and integration tests (8 test files)
├── docs/                 # Design documentation
│   ├── DESIGN.md         # Complete architecture design (15 chapters)
│   ├── ROADMAP_TO_SEEDANCE_2_5.md  # Scale-up roadmap: data → training → deployment
│   └── GLOBAL_DEPLOYMENT_AND_BILLING.md  # Global deployment + billing system design
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.4+
- CUDA-capable GPU (24GB+ VRAM recommended)

## Documentation

Detailed design documentation is available in [docs/](docs/):

| Document | Description |
|----------|-------------|
| [DESIGN.md](docs/DESIGN.md) | Complete architecture design — 15 chapters covering all components, physics consistency, loss functions, and performance benchmarks |
| [ROADMAP_TO_SEEDANCE_2_5.md](docs/ROADMAP_TO_SEEDANCE_2_5.md) | Scale-up roadmap — data preparation, annotation pipeline, distributed pre-training, post-training, and infrastructure planning |
| [GLOBAL_DEPLOYMENT_AND_BILLING.md](docs/GLOBAL_DEPLOYMENT_AND_BILLING.md) | Global deployment design — multi-region architecture, billing system, multi-tenancy, security compliance, and cost optimization |

## References

**Seedance Architecture:**
- [Seedance 2.0: Advancing Video Generation for World Complexity](https://arxiv.org/abs/2604.14148) — ByteDance Seed Team, 2026
- [Seedance 1.5 Pro: A Native Audio-Visual Joint Generation Foundation Model](https://arxiv.org/abs/2512.13507) — ByteDance Seed Team, 2025

**Diffusion & Flow Matching:**
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., 2023
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Esser et al., 2024

**Physics Consistency:**
- [CausalMotion: Structured Physical Reasoning as Keyframe and Trajectory Guidance](https://arxiv.org/abs/2606.xxxxx) — Zhuang et al., 2026
- [Physics in 2-Steps: Locking Motion Priors Before Visual Refinement Erases Them](https://arxiv.org/abs/2606.xxxxx) — Han et al., ICML 2026
- [Enhancing Video Physical Consistency via DPO](https://arxiv.org/abs/2512.xxxxx) — Wang et al., 2025
- [The Invisible Hand of Physics in Video Diffusion Models](https://arxiv.org/abs/2606.xxxxx) — Esmati et al., 2026
- [Enhancing Video Physical Consistency via Role-aware Joint Training](https://arxiv.org/abs/2607.xxxxx) — Zheng et al., 2026

**Infrastructure & Scaling:**
- [Open-Sora: Democratizing Efficient Video Production](https://github.com/hpcaitech/Open-Sora) — HPC-AI Tech
- [LongCat-Video: Cascaded Coarse-to-Fine for Minute-Long Generation](https://arxiv.org/abs/2510.xxxxx) — 2025

## License

MIT
