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

The training data pipeline is designed to produce a **balanced, caption-rich** dataset
with diverse human portraits, talking faces, and general visual scenes.

#### Overview

```
┌─────────────────┐   ┌───────────────┐   ┌──────────────┐   ┌─────────────┐
│ 1. Download      │──▶│ 2. Filter     │──▶│ 3. Caption   │──▶│ 4. Build     │
│    per-source    │   │    quality    │   │    VLM/template│   │    manifest  │
└─────────────────┘   └───────────────┘   └──────────────┘   └─────────────┘
```

Final training manifest: **36k videos** with **13k unique captions**, 75% high-quality faces.

---

#### Step 1: Download Data Sources

Each source is downloaded independently via dedicated tools in `seedance/tools/` and `scripts/`.

**1a. CelebA-HQ (30,000 high-res face images — primary source)**

```bash
# Download from HuggingFace (requires proxy setup if in mainland China)
# Model: Ryan-sjtu/celebahq-caption (images + captions in parquet format)
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
hf download --repo-type dataset Ryan-sjtu/celebahq-caption --local-dir data/celeba_hq

# Extract parquet → pseudo-videos with subtle crop jitter for motion tolerance
python scripts/extract_celeba_hq.py
# Output: 30,000 pseudo-videos (32f, 256×256, 3.4 GB)
# Manifest: data/manifests/celeba_hq_train.csv (28,500 train + 1,500 val)
```

**1b. Pexels (stock footage)**

```bash
python -m seedance.tools.download_pexels \
    --categories people,animals,city,food,nature,tech,travel \
    --output data/pexels_raw/ \
    --max_per_category 200

# Quality filter: resolution ≥ 360p, duration 2-120s, optical flow 0.05-8.0
python -m seedance.tools.quality_filter \
    --input data/pexels_raw/ \
    --output data/pexels_filtered/ \
    --min_height 360 --min_duration 2.0 --max_duration 120
```

**1c. WebVid (10M web videos with captions)**

```bash
python -m seedance.tools.download_webvid \
    --num_videos 10000 --output data/webvid/

# Quality filter
python -m seedance.tools.quality_filter \
    --input data/webvid/videos/ \
    --output data/webvid_filtered/
```

**1d. Bilibili (Chinese video platform — portrait/fashion content)**

```bash
# Download portrait/fashion videos by keyword search
python scrapers/bilibili_download.py \
    --tags "街拍穿搭,近距离人像,原相机拍摄,半身人像,自然光人像,生活vlog,少女写真" \
    --output data/people_bilibili \
    --max 3000 --workers 4

# Quality filter with optical flow
python -m seedance.tools.quality_filter \
    --input data/people_bilibili \
    --output data/people_bilibili_filtered \
    --min_height 360 --max_duration 120
```

**1e. VoxCeleb2 (talking faces, ~1.1M clips)**

Download from [robots.ox.ac.uk/~vgg/data/voxceleb/](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/).
Extract with 7-Zip, then generate manifest:

```bash
python -m seedance.tools.ingest_talking_data \
    --input_dir data/voxceleb2/dev/mp4/ \
    --dataset voxceleb \
    --output data/manifests/voxceleb_manifest.csv
```

---

#### Step 2: VLM Captioning (Qwen2-VL)

Template captions ("A person speaking") don't teach the model text-to-content
mapping. Real descriptive captions from a vision-language model (VLM) do.

**Which videos need captioning:** Any video without a real descriptive caption —
pexels_people, bilibili_people, filtered_people. WebVid videos have Shutterstock
captions already. CelebA-HQ comes with captions.

```bash
# Install Qwen2-VL dependencies (in project venv)
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install qwen-vl-utils

# Run captioning (requires ~15 GB GPU memory for Qwen2-VL-2B)
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
.venv/bin/python scripts/caption_videos_vlm.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --input data/pexels_people/ data/people_bilibili_filtered/ data/filtered/people/ \
    --output data/manifests/vlm_captions.json \
    --num_frames 4 \
    --resume
```

**How it works:**
1. Sample 4 evenly-spaced frames per video, resize to max 448px
2. Feed to Qwen2-VL-2B with a structured prompt asking for person description
3. Each video takes ~0.6s, ~10 minutes for 1,000 videos
4. Output saved to `vlm_captions.json`, auto-resume on failure

**Caption quality examples:**

| Before (template) | After (VLM) |
|---|---|
| "a young woman walking in a bright room, well-composed" | "A young woman with long brown hair smiling at the camera in soft natural window light, medium close-up portrait" |
| "A person sitting in a contemporary setting" | "A young man with short black hair and a beard, wearing a black tank top and blue jeans, sitting in a cafe with a window in the background" |

> **Model choice:** `Qwen/Qwen2-VL-2B-Instruct` (~4 GB) for speed. Swap to
> `Qwen/Qwen2-VL-7B-Instruct` (~14 GB) for higher quality captions.
>
> **Alternative:** For production-scale captioning, consider
> [CogVLM2-Video](https://huggingface.co/THUDM/cogvlm2-video-llama3-chat)
> which understands temporal dynamics better (but requires ~40 GB GPU memory).

---

#### Step 3: Build Balanced Training Manifest

The manifest builder merges all sources into a single train/val split with
controlled proportions.

```bash
python scripts/build_balanced_manifest.py \
    --vox_count 3000 \       # VoxCeleb2: limit to 3k (from 1.1M)
    --general_count 1500 \    # General non-people videos
    --output data/manifests/train_balanced.csv
```

**What it does:**
1. Loads all video directories, parquet manifests, and VLM captions
2. VLM captions take priority over template captions (auto-detected)
3. Downsamples VoxCeleb2 to `--vox_count` (default 15,000)
4. Adds all people-related videos (CelebA-HQ, pexels_people, bilibili_people, webvid_people)
5. Adds general scenes (WebVid, Pexels categories) for visual diversity
6. Shuffles and splits into train/val (95/5 default)

---

#### Final Training Data Composition

| Data Source | Videos | % | Resolution | Caption Source | Description |
|-------------|--------|---|-----------|----------------|-------------|
| **CelebA-HQ** | 27,061 | 75.2% | 256×256 | Original (8.4k unique) | High-res face portraits |
| WebVid people | 3,051 | 8.5% | 596×336 | Shutterstock | General people scenes |
| VoxCeleb2 | 2,867 | 8.0% | 224×224 | 14 diverse templates | Talking faces (downsampled) |
| Bilibili people | 554 | 1.5% | 720×1280 | **Qwen2-VL** | Portrait/fashion videos |
| pexels_people | 162 | 0.5% | 1080p-4K | **Qwen2-VL** | High-res people stock |
| General + animation | ~2,296 | 6.3% | Mixed | Shutterstock / templates | Visual diversity |
| **Total** | **~36,000** | **100%** | | **13k unique captions** | |

> See [scripts/build_balanced_manifest.py](scripts/build_balanced_manifest.py) for
> full configuration options and `--help` for usage.

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
