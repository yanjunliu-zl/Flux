# From Seedance 2.0 to 2.5: Engineering Roadmap for Scale

> This article analyzes the engineering work required to advance the current open-source Seedance 2.0 reference implementation to a production-grade Seedance 2.5 system, covering six areas: data, annotation, pre-training, post-training, distributed deployment, and infrastructure.

---

## 1. Target Definition: What Is Seedance 2.5

Seedance 2.5 capability baseline:

| Dimension | 2.0 (Current) | 2.5 (Target) |
|-----------|--------------|--------------|
| **Model Scale** | 1.6B dense, 30B MoE configs | 200B MoE main model + cascade sub-models |
| **Output Resolution** | 256×256 training, 512×512 fine-tuning | Native 4K (3840×2160) |
| **Output Duration** | 2-4s (32fr @ 16fps) | 30s (128fr latent @ 30fps) |
| **Input Conditions** | Text / single image | 50 reference inputs (multi-image + video + audio + pose + depth) |
| **Training Data** | ~1.1M videos, ~265 GB | 1B+ videos, PB-scale |
| **Training Compute** | Single node 8×A100 | Thousand-GPU cluster, tens of thousands GPU·hours |
| **Text Encoder** | T5-XXL (4.7B) | T5-XXL + multi-modal encoder |
| **Physics Consistency** | 6 mechanism prototypes | Full pipeline productionized |

---

## 2. Data Preparation

### 2.1 Data Scale

The current VoxCeleb2 + WebVid + HDTF (~1.1M videos, 265 GB) is a proof-of-concept scale. Seedance 2.5 requires:

| Data Type | Target Volume | Sources | Challenges |
|-----------|-------------|---------|------------|
| **General Videos** | 500M–1B clips | WebVid-10M, HD-VILA-100M, internal crawlers | Copyright cleaning, NSFW filtering, dedup |
| **High-Quality Videos** | 5–10M clips | Stock footage (Shutterstock licensed), film clips | Copyright licensing, metadata standardization |
| **Talking Faces** | 20–50M clips | VoxCeleb2 + expanded crawlers + podcast videos | Multi-language coverage, accent diversity |
| **Physical Interactions** | 5–10M clips | IntPhys, Something-Something, sports footage | Physics annotation difficulty, scene diversity |
| **4K Videos** | 1–5M clips | 4K YouTube, professional cinematography | Download bandwidth, transcoding overhead |
| **Multi-modal Pairs** | 10–30M clips | AudioSet, VGGSound, internal AV data | Audio-video alignment precision |

### 2.2 Crawling Infrastructure

Building a large-scale video crawling system from scratch:

```
                           ┌─────────────┐
                           │  URL Seed DB │
                           │  (YouTube,   │
                           │   Pexels,    │
                           │   TikTok...) │
                           └──────┬──────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ yt-dlp   │ │ Custom   │ │ API      │
              │ Worker   │ │ Scraper  │ │ Worker   │
              │ (YouTube)│ │ (Web)    │ │ (Pexels) │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   └────────────┼────────────┘
                                ▼
                    ┌──────────────────────┐
                    │   Download Queue      │
                    │   (Redis)            │
                    │   Priority: quality   │
                    │   over quantity       │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Distributed Download │
                    │  100-500 Worker Nodes │
                    │  1-10 Gbps per node   │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Raw Video Storage    │
                    │  Object Store (S3)    │
                    │  Expected: 5-20 PB    │
                    └──────────────────────┘
```

**Key Requirements**:
- **Speed control**: Adaptive rate limiting to avoid triggering platform anti-scraping
- **Format normalization**: Unified transcode to H.264/H.265, standardized FPS and resolution
- **Resumable downloads**: Chunked downloads for large files (>1GB)
- **Dedup pipeline**: Perceptual hash (pHash) + content fingerprinting
- **Legal compliance**: Copyright detection + robots.txt adherence + regional compliance
- **Cost estimation**: Crawling 100M videos requires ~300-500 node·months, bandwidth ~$50K-200K/month

### 2.3 Data Storage & Format

| Tier | Format | Compression | Purpose | Estimated Capacity |
|------|--------|------------|---------|-------------------|
| **Raw Video** | MP4/MKV | H.264/H.265 | Permanent archive | 5-20 PB |
| **Preprocessed Frames** | Zarr/NPZ | None (fast random access) | Training hot data | 500 TB – 2 PB |
| **Latent Cache** | Safetensors | None | VAE-encoded cache | 50-200 TB |
| **Annotation Index** | Parquet + Lance | Zstd | Fast query and filtering | 10-50 TB |
| **Embedding Index** | FAISS/Lance | None | Similarity search and dedup | 5-20 TB |

**Zarr Sharding Strategy** (for fast random access during training):
```
data/zarr/
  video/
    shard_000000/
      chunk_0000.zarr   ← 1024 videos per chunk, (T, H, W, C) uint8
      chunk_0001.zarr
      ...
  audio/
    shard_000000/
      chunk_0000.zarr   ← 1024 audios per chunk, (T_mel, n_mels) float16
```

**Advantages**: Zarr supports multi-threaded parallel reading, sharded random access, seamless PyTorch DataLoader integration, and direct streaming from object storage.

### 2.4 Data Quality System

```
Raw Video
  │
  ├── [1] Technical Quality Filter
  │   ├── Resolution ≥ 360p
  │   ├── FPS ≥ 12fps
  │   ├── Duration ≥ 3s (exclude shorts/GIFs)
  │   ├── Bitrate ≥ 500kbps
  │   ├── No corrupted frame detection
  │   └── Non-monochrome/solid-color detection
  │
  ├── [2] Content Quality Filter
  │   ├── Blur detection (Laplacian variance)
  │   ├── Overexposure/underexposure detection
  │   ├── Severe shake detection (global motion estimation)
  │   ├── Scene cut frequency reasonability
  │   └── Watermark/subtitle/logo detection
  │
  ├── [3] Safety Filter
  │   ├── NSFW image detection (NudeNet/CLIP-based)
  │   ├── Violent content detection
  │   ├── Child protection compliance
  │   └── Copyright watermark identification
  │
  ├── [4] Diversity Analysis
  │   ├── Scene type distribution (indoor/outdoor/nature/urban…)
  │   ├── Motion type distribution (fast/slow/static/interactive…)
  │   ├── Lighting condition distribution
  │   └── Cultural/geographic diversity
  │
  └── [5] Deduplication
      ├── Exact dedup (MD5/SHA256)
      ├── Near-dedup (pHash Hamming distance ≤ 5)
      ├── Semantic dedup (CLIP embedding cosine similarity > 0.98)
      └── Cross-source dedup (same video uploaded to multiple platforms)
```

**Quality Filter Pass Rates**:
| Stage | Pass Rate | 100M raw → |
|-------|----------|-----------|
| Technical quality | 85% | 85M |
| Content quality | 60% | 51M |
| Safety filter | 90% | 45.9M |
| Deduplication | 70% | 32.1M |
| **Final usable** | **~32%** | **~32M** |

---

## 3. Annotation Pipeline

### 3.1 Annotation Dimensions

Seedance 2.5 requires far deeper annotation than 2.0:

```
Video Input
  │
  ├── [Basic Annotation] ─────────────────────────────────────
  │   ├── Short caption (1-2 sentences, human+auto hybrid)
  │   ├── Long caption (paragraph-level, Video-LLaMA-2/CogVLM2)
  │   ├── Timestamped captions (event descriptions every 2s)
  │   └── Multi-language captions (EN/ZH/JA/KO coverage)
  │
  ├── [Structural Annotation] ─────────────────────────────────
  │   ├── Scene type (100+ classes)
  │   ├── Shot type (wide/medium/close-up/…)
  │   ├── Camera motion (static/pan/zoom/tracking/…)
  │   ├── Lighting condition (natural/indoor/backlit/…)
  │   └── Color tone/style (realistic/animated/vintage/…)
  │
  ├── [Physics Annotation] ────────────────────────────────────
  │   ├── Physics event type (collision/gravity/fluid/contact/… 8 classes)
  │   ├── Object physical role (agent/controlled/passive)
  │   ├── Motion trajectory (key object 2D trajectory)
  │   └── Causal chain (A pushes B → B hits C → C falls)
  │
  ├── [Character Annotation] ──────────────────────────────────
  │   ├── Face detection + identity clustering
  │   ├── 3D keypoints (MediaPipe/InsightFace)
  │   ├── Expression classification (7 basic + neutral)
  │   ├── Mouth/viseme annotation (LipSync training data)
  │   ├── Body pose (OpenPose/DWPose)
  │   └── Gesture recognition
  │
  ├── [Audio Annotation] ──────────────────────────────────────
  │   ├── Speech transcription (Whisper large-v3)
  │   ├── Audio event classification (AudioSet 527 classes)
  │   ├── AV alignment offset (onset detection)
  │   ├── Music/SFX/speech separation (Demucs/HTDemucs)
  │   └── Emotion/tone (speech emotion recognition)
  │
  ├── [Deep Annotation] ───────────────────────────────────────
  │   ├── Depth estimation (DepthAnything-V2, per-frame)
  │   ├── Optical flow (RAFT/GMA, per adjacent frame pair)
  │   ├── Semantic segmentation (SAM-2, keyframes)
  │   ├── Instance segmentation (object tracking ID)
  │   └── Surface normal estimation
  │
  └── [Quality Scoring] ───────────────────────────────────────
      ├── Visual quality (BRISQUE/NIQE/CLIP-IQA)
      ├── Motion smoothness
      ├── Composition aesthetics (AVA dataset scoring)
      └── Physical plausibility (PhysicsEventDetector)
```

### 3.2 Annotation Compute Cost

Annotation is the most compute-intensive part of the pipeline. Estimated for 32M videos (avg 10s):

| Annotation Task | Model | Per-Video Time | Total GPU·h (32M) | GPU Needed |
|----------------|-------|---------------|-------------------|-----------|
| **Short caption** | BLIP-2 / Video-LLaMA-2 | ~5s | 44,400 h | 10×A100 |
| **Long caption** | CogVLM2-19B | ~15s | 133,200 h | 30×A100 |
| **Depth estimation** | DepthAnything-V2 | ~3s/frame, 8 frames sampled | 213,300 h | 50×A100 |
| **Optical flow** | RAFT | ~0.2s/frame pair | 56,800 h | 15×A100 |
| **Segmentation** | SAM-2 | ~1s/keyframe | 8,900 h | 2×A100 |
| **Face + keypoints** | InsightFace | ~0.1s/frame | 28,400 h | 8×A100 |
| **Speech transcription** | Whisper large-v3 | ~2s | 17,800 h | 4×A100 |
| **Audio events** | PANNs/CLAP | ~1s | 8,900 h | 2×A100 |
| **Physics events** | PhysicsEventDetector (CPU) | ~0.02s | 178 h | CPU only |
| **Quality scoring** | BRISQUE+NQI+CLIP | ~3s | 26,700 h | 6×A100 |
| **Safety filter** | NudeNet+CLIP NSFW | ~1s | 8,900 h | 2×A100 |
| **Total** | | | **~547,500 GPU·h** | **~130×A100** |

With 130 A100 GPUs running continuously, the annotation pipeline takes **~6 months** to complete all 32M videos. Using H100 reduces this to ~3 months.

### 3.3 Annotation Pipeline Architecture

```
                         ┌──────────────────────┐
                         │  Annotation Scheduler │
                         │  (Argo)              │
                         │  DAG: dependency mgmt │
                         │  Priority queue       │
                         └──────────┬───────────┘
                                    │
          ┌─────────────┬───────────┼───────────┬─────────────┐
          ▼             ▼           ▼           ▼             ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Stage 1   │ │Stage 2   │ │Stage 3   │ │Stage 4   │ │Stage 5   │
    │Tech      │ │Content   │ │Deep      │ │Character │ │Quality   │
    │Filter    │ │Annotate  │ │Annotate  │ │Annotate  │ │Score     │
    │          │ │          │ │          │ │          │ │          │
    │CPU Worker│ │GPU Worker│ │GPU Worker│ │GPU Worker│ │GPU Worker│
    │100×      │ │30×A100   │ │50×A100   │ │10×A100   │ │10×A100   │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Annotation Database  │
                         │  Lance/Parquet        │
                         │  Versioned Schema     │
                         │  Incremental updates  │
                         └──────────────────────┘
```

**Key Design Points**:
- **DAG scheduling**: Stage 1 output triggers Stage 2-5 (parallelizable)
- **Incremental processing**: New data auto-triggers annotation, no full re-runs
- **Schema versioning**: Backward compatible when annotation dimensions increase
- **Quality spot-checking**: 1% human review + auto-annotation confidence thresholds

---

## 4. Pre-Training

### 4.1 Training Scale Estimation

The scale jump from 1.6B dense to 200B MoE:

| Parameter | 1.6B Base (Current) | 30B Dense | 200B MoE (2.5 Target) |
|-----------|--------------------|-----------|----------------------|
| dim | 1024 | 2048 | 4096 |
| num_layers | 24 | 48 | 48-72 |
| num_heads | 16 | 32 | 32-48 |
| Total params | 1.6B | 30.6B | ~200B |
| Activated params | 1.6B (100%) | 30.6B (100%) | ~36B (18%) |
| Per-step VRAM (bf16) | ~20 GB | ~130 GB | ~500 GB |
| Minimum GPU | 1×A100 40GB | 4×A100 80GB | 8×H100 80GB |
| Recommended GPU | 8×A100 80GB | 8×H100 80GB | 32×H100 80GB |
| Batch per GPU | 8 | 4 | 1 |
| Gradient accumulation | 2 | 8 | 64 |
| Effective batch | 128 | 128 | 512 |
| Training steps | 500K | 500K | 1M |
| Total tokens | ~1.6T | ~6.4T | ~256T |
| **GPU·hours** | **~11,000** | **~55,000** | **~3,000,000** |

200B MoE training requires a 32×H100 cluster running for **~4 months** (assuming 80% utilization).

### 4.2 Distributed Training Strategy

#### 4.2.1 Hybrid Parallelism

For 200B-scale models, a single parallelism strategy is insufficient — hybrid parallelism is required:

```
Hybrid Parallelism:
  ┌─────────────────────────────────────────────────────────────┐
  │  Data Parallel (DP): 4-way — across nodes                   │
  │  Fully Sharded (FSDP/ZeRO-3): params+grads+optimizer sharded│
  │  Tensor Parallel (TP): 2-way — within-node NVLink           │
  │  Pipeline Parallel (PP): 4-way — inter-layer pipelining     │
  │  Sequence Parallel (SP): 8-way — Ulysses SP (long sequences)│
  │  Expert Parallel (EP): 16-way — MoE experts across nodes    │
  └─────────────────────────────────────────────────────────────┘

Recommended config: 32×H100 (4 nodes × 8 GPU)
  DP=4, TP=2, PP=4, EP=16
  → Per GPU ~22B params + optimizer states ≈ 60-70 GB
  → bf16 training feasible, no CPU offload needed
```

#### 4.2.2 Communication Optimization

| Technique | Purpose | Expected Gain |
|-----------|---------|---------------|
| **FSDP prefetch** | Preload next layer params, hide communication latency | 10-15% throughput |
| **Gradient bucketing** | Merge small gradient tensors, reduce all-reduce count | 5-10% comm reduction |
| **Communication overlap** | Overlap compute and communication | 15-25% throughput |
| **FP8 communication** (H100) | All-reduce using FP8 reduces bandwidth | 2× comm bandwidth |
| **SHARP** (InfiniBand) | In-network aggregation, reduce data roundtrips | 30-50% comm reduction |
| **NVLink Switch** | 900 GB/s full interconnect within node | Near-zero TP overhead |

#### 4.2.3 DeepSpeed Configuration Evolution

Current code has DeepSpeed support but needs extension to 200B scale:

```python
# 200B MoE DeepSpeed Config (to be added)
DEEPSPEED_200B_CONFIG = {
    "train_batch_size": 512,
    "gradient_accumulation_steps": 64,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "stage3_max_live_parameters": 3e9,
        "stage3_max_reuse_distance": 3e9,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "nvme",                    # NVMe offload for 200B
            "nvme_path": "/mnt/nvme/deepspeed",
            "buffer_count": 8,
            "buffer_size": 1e9,
        },
    },
    "expert_parallel_size": 16,
    "moe": {
        "ep_size": 16,
        "enable_expert_tensor_parallelism": True,
    },
    "sequence_parallel_enabled": True,
    "communication_data_type": "fp16",
    "prescale_gradients": True,
    "gradient_clipping": 1.0,
}
```

### 4.3 Training Stability

Stability is the primary challenge for large model training:

| Problem | Symptom | Mitigation |
|---------|---------|------------|
| **Loss spike** | Loss suddenly spikes 10-100× | Gradient clipping (1.0), skip anomalous batches, rollback checkpoint |
| **Router collapse** | MoE routing degenerates to single expert | Load balancing loss, expert dropout, router z-loss |
| **Activation explosion** | Intermediate activations overflow | QK norm, AdaLN normalization, mixed precision NaN detection |
| **Slow convergence** | Training loss decreases slowly | μP (maximal update) parameterization, LR warmup |
| **Dead experts** | Some experts never get routed | Expert re-initialization strategy, load balance warmup |
| **Gradient noise** | Poor gradient quality under large batches | Gradient accumulation, adaptive batch size, LAMB optimizer |
| **Checkpoint corruption** | State inconsistency on save/load | Checksum verification, atomic writes, backup checkpoint |

### 4.4 Multi-Stage Training Curriculum

Seedance 2.5 training is more complex than 2.0:

```
Stage 0 — Text Encoder Warmup (optional)
  ├── Freeze T5-XXL, train cross-attn projection only
  ├── Data: 100M image-text pairs
  └── Steps: 50K

Stage 1 — Video Pretraining (500K steps)
  ├── Vision branch: PixArt-α init → train
  ├── Audio branch: frozen
  ├── CBGA: frozen
  ├── Data: 500M+ general videos
  ├── Resolution: 256×256, 16-32 frames
  └── Loss: Flow Matching (video only)

Stage 2 — Audio Pretraining (200K steps)
  ├── Vision branch: frozen
  ├── Audio branch: trained from scratch
  ├── Data: 100M+ audio + paired videos
  └── Loss: Flow Matching (audio only)

Stage 3 — AV Joint Training (300K steps)
  ├── Full model training, CBGA active
  ├── Data: 30M AV pairs
  ├── Resolution: 256×256, 16-32 frames
  └── Loss: Flow Matching + Sync + World Model + VPT

Stage 4 — Physics Preference Optimization (100K steps)
  ├── PhyDPO training
  ├── PhysicsRM scoring → preference pairs
  └── Data: 5M physics interaction videos

Stage 5 — High-Resolution Fine-Tuning (100K steps)
  ├── Resolution: 512×512, 64 frames
  ├── Progressive resolution increase: 256→384→512
  └── Data: 1M high-quality HD videos

Stage 6 — Long-Sequence Adaptation (50K steps)
  ├── Frames: 64→96→128
  ├── NTK RoPE scaling
  ├── Sparse attention switch
  └── Data: 500K long videos (≥30s)

Stage 7 — SFT Supervised Fine-Tuning (50K steps)
  ├── LFA + KP + shot control
  ├── Data: 50-100K curated shot-level annotations
  └── High aesthetic filter

Stage 8 — RLHF PPO (10K steps)
  ├── 5-D reward model
  ├── Best-of-N + PPO clip
  └── Data: 10K human preference annotations
```

**Total Training Time**: Stage 1-6 ≈ 4-5 months (32×H100), Stage 7-8 ≈ 2-3 weeks.

---

## 5. Post-Training

### 5.1 Reward Model Training

**Data requirements**: At least 10,000 human preference annotations. Each annotation = 2-4 generations for the same prompt → humans score across 5 dimensions.

| Dimension | Scoring Criteria | Annotation Difficulty |
|-----------|-----------------|----------------------|
| visual_quality | Sharpness, color, detail | Low |
| motion_smoothness | Motion fluidity, no jitter | Medium |
| character_consistency | Cross-frame character identity | Medium |
| av_sync | Audio-visual sync, lip matching | High (needs expert review) |
| prompt_alignment | Semantic alignment | Low |
| **physics_plausibility** (new) | Motion/collision/gravity plausibility | High (needs physics intuition) |

**Annotation Flow**:
```
prompt → generate 4 candidates
  → 5 annotators score independently (1-5 Likert)
  → Compute Krippendorff's alpha (inter-annotator agreement)
  → alpha ≥ 0.7 → include in training set
  → alpha < 0.7 → submit to senior reviewer for arbitration
```

**RM Architecture Enhancement**:
- Current: Shared 3D Conv Backbone → 5 independent heads
- 2.5 upgrade: Add Physics Head (distilled from PhysicsRM), Add Aesthetic Head (AVA dataset pretrained)

### 5.2 SFT Data Construction

**Shot-level Annotation** is the core data requirement for SFT:

```
A high-quality SFT sample:
{
  "video": "shot_#12345.mp4",
  "prompt": "Camera slowly pushes from wide shot to close-up of the character's face, who smiles and turns toward the camera",
  "shot_type": "dolly_push_in",
  "shot_scale": ["wide", "medium", "close_up"],
  "camera_motion_path": [
    {"t": 0.0, "pos": [0, 0, 10], "look_at": [0, 1.5, 0]},
    {"t": 1.0, "pos": [0, 0.5, 2], "look_at": [0, 1.5, 0]},
  ],
  "character_id": "char_A",
  "facial_kp_3d": "kp_#12345.npy",
  "audio_event": "speech",
  "viseme_sequence": [0, 4, 11, 4, ...],
  "dialogue_transcript": "I've been waiting for this moment.",
  "physics_events": ["arm_lift", "head_turn"],
  "lighting": "three_point_warm",
  "aesthetic_score": 8.2,
}
```

**Data volume**: 50-100K manual annotations + 500K-1M auto-annotated (hybrid strategy).

### 5.3 Safety Alignment

Video generation safety is more complex than text generation:

| Risk Category | Detection Method | Mitigation |
|--------------|-----------------|------------|
| **Deepfake faces** | Face matching detection + liveness detection | Limit face generation scope, add invisible watermark |
| **Violence/gore** | Video content safety classifier | Inference-time safety prompt injection + output classifier filtering |
| **Child-unsafe content** | CSAM hash matching + age classifier | Training data cleaning + output review |
| **Copyright infringement** | Output vs. training set similarity detection | Generation diversity constraints + copyright watermark |
| **Misinformation** | Scene-text consistency verification | Synthetic video labeling (C2PA standard) |

**Safety alignment technical route**:
1. **Training phase**: Thoroughly remove high-risk content from training data
2. **SFT phase**: Add refusal samples ("generate a cat" → normal, "generate violent scene" → safe refusal)
3. **RLHF phase**: Safety reward signal (negative reward = probability of unsafe content)
4. **Inference phase**: Input/output dual safety classifier + invisible watermark (StegaStamp)

---

## 6. Inference Deployment

### 6.1 Cascaded Inference Pipeline

Seedance 2.5 inference cannot be done in one shot — requires cascading:

```
User Input: prompt + reference images/video (up to 50)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage A: Text Understanding + Planning                      │
│ T5-XXL encode + VLM prompt decomposition + keyframe planning│
│ Time: ~2-5s, 1×A100                                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage B: Coarse — Structural Generation                     │
│ 32fr × 256×256, Flow Matching 30 steps, Euler              │
│ Time: ~15s, 1×A100                                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage C: Temporal Extension                                 │
│ 32fr → 128fr (30s), NTK RoPE + sparse attention, 10 steps  │
│ Time: ~30s, 1×A100                                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage D: Spatial SR — Spatial Super-Resolution              │
│ 256→512→1024→4K, Cascaded upsampling + 10 steps each       │
│ Time: ~90s, 2×A100 (tiled)                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage E: Audio Generation                                   │
│ Audio latent → Flow Matching → AudioVAE decode → waveform  │
│ Time: ~20s, 1×A100                                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage F: Post-processing                                    │
│ AV mux → Color grading → Watermark → Encode (H.265)        │
│ Time: ~10s, CPU                                             │
└─────────────────────────────────────────────────────────────┘

Total: ~3 min / 4K 30s video (3-4×A100)
```

### 6.2 Inference Optimization Techniques

| Technique | Applicable Stage | Speedup | Difficulty |
|-----------|-----------------|---------|------------|
| **TensorRT/ONNX compilation** | DiT backbone | 1.5-2× | Medium |
| **FP8/INT8 quantization** | Attention + FFN | 1.5-2× | Medium |
| **FlashAttention-3** (H100) | All attention | 1.5-2× | Low (swap backend) |
| **vLLM-style PagedAttention** | KV cache management | 1.2× | High |
| **Torch.compile** | Full model | 1.3-1.5× | Low |
| **Diffusion distillation** (LCM/ADD) | Flow Matching sampling | 5-10× (4-8 steps) | High (extra training) |
| **Step distillation** | 30 steps → 4 steps | 7.5× | High |
| **Tiled VAE decoding** | Spatial SR stage | 2-3× (VRAM) | Low |
| **Sparse attention** | Temporal stage | 2-4× (long seqs) | Already implemented |

### 6.3 Online Service Architecture

```
                         ┌─────────────┐
                         │  API Gateway │
                         │  (Rate limit,│
                         │   Auth,      │
                         │   Queue)     │
                         └──────┬──────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌──────────────┐         ┌──────────────┐
          │ Task Scheduler│         │  Result Cache │
          │ (Celery/K8s) │◄───────►│  (Redis)      │
          │ Priority Queue│         │  Same prompt   │
          └──────┬───────┘         │  cache reuse   │
                 │                 └──────────────┘
     ┌───────────┼───────────┬──────────────┐
     ▼           ▼           ▼              ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
│GPU Pod  │ │GPU Pod  │ │GPU Pod  │ │CPU Pod   │
│Stage B  │ │Stage C  │ │Stage D  │ │Stage E+F │
│A100×1   │ │A100×1   │ │A100×2   │ │CPU×4     │
│Coarse   │ │Temporal │ │Spatial  │ │Audio+Post│
└─────────┘ └─────────┘ └─────────┘ └──────────┘
     │           │           │              │
     └───────────┴───────────┴──────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Object Store (S3) │
          │ Output + Metadata │
          └──────────────────┘
```

**SLA Targets**: P50 latency < 3 min, P99 latency < 10 min, availability 99.5%.

**Auto-scaling**: Based on GPU queue depth (KEDA + K8s HPA).

**Cost Estimate** (AWS on-demand):
| Stage | GPU Type | Count | Hourly Cost | Per-Video Cost |
|-------|---------|-------|-------------|----------------|
| B: Coarse | A100 80GB | 1 | $3.06 | $0.08 |
| C: Temporal | A100 80GB | 1 | $3.06 | $0.15 |
| D: Spatial SR | A100 80GB | 2 | $6.12 | $0.46 |
| E+F: Audio+Post | CPU | 4 | $0.40 | $0.02 |
| **Total** | | | **$12.64/h** | **~$0.71/video** |

---

## 7. Infrastructure

### 7.1 Compute Cluster

```
┌─────────────────────────────────────────────────────────────────┐
│                    Compute Cluster Topology                      │
│                                                                  │
│  ┌──────────────────────────┐    ┌──────────────────────────┐   │
│  │  Training (Dedicated)    │    │  Inference (Elastic)     │   │
│  │                          │    │                          │   │
│  │  32×H100 80GB SXM        │    │  8-32×A100 80GB         │   │
│  │  4 nodes × 8 GPU         │    │  K8s GPU Pods            │   │
│  │  NVLink Switch 900GB/s   │    │  Auto-scaling            │   │
│  │  InfiniBand NDR400       │    │  Spot/on-demand mix      │   │
│  │  Shared parallel FS      │    │  Hot model loading       │   │
│  └──────────────────────────┘    └──────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────┐    ┌──────────────────────────┐   │
│  │  Annotation (Elastic)    │    │  Storage Cluster         │   │
│  │                          │    │                          │   │
│  │  50-130×A100 80GB        │    │  WekaFS / Lustre         │   │
│  │  GPU time-sharing        │    │  Training: 1-2 PB NVMe   │   │
│  │  CPU spot (filtering)    │    │  Archive: 10-20 PB HDD   │   │
│  │                          │    │  Object store: S3 compat │   │
│  └──────────────────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Storage Architecture

| Tier | Technology | Capacity | IOPS/Throughput | Purpose |
|------|-----------|----------|-----------------|---------|
| **Hot** (NVMe) | WekaFS / DDN A³I | 500 TB – 1 PB | 100+ GB/s read | Training data, checkpoints |
| **Warm** (SSD) | MinIO / Ceph | 2-5 PB | 20 GB/s | Intermediate results, annotation cache |
| **Cold** (HDD) | MinIO / Tape | 10-20 PB | 5 GB/s | Raw video archive |
| **Metadata** | PostgreSQL + Redis | 100 GB | — | Data index, training metadata |

### 7.3 Network Topology

```
Training Nodes (4× H100 nodes)
  ├── Intra-node: NVLink Switch 900 GB/s (GPU-GPU)
  ├── Inter-node: InfiniBand NDR400 400 Gbps (RDMA)
  │   └── Fat-tree topology, no oversubscription
  └── Storage: 4× 200 Gbps storage network

Inference Nodes (8× A100 nodes)
  ├── Intra-node: NVLink 600 GB/s
  ├── Inter-node: 100 Gbps RoCE
  └── Storage: 2× 100 Gbps

Annotation Nodes
  ├── GPU nodes: 25 Gbps
  └── CPU nodes: 10 Gbps
```

### 7.4 Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────┐
│                  Monitoring Stack                            │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Metrics      │  │ Logging      │  │ Tracing             │  │
│  │ Prometheus   │  │ Loki +       │  │ OpenTelemetry       │  │
│  │ + Grafana    │  │ Grafana      │  │ + Jaeger            │  │
│  └──────┬───────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                 │                     │             │
│         ▼                 ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Key Metrics:                                             │ │
│  │ GPU utilization, MFU (Model FLOPs Utilization)            │ │
│  │ Communication bandwidth utilization, All-reduce latency   │ │
│  │ Loss / Gradient Norm / LR                                │ │
│  │ MoE Router distribution (expert utilization)              │ │
│  │ Training throughput (tokens/s, samples/s)                │ │
│  │ Storage IOPS, Network packet loss                         │ │
│  │ Node temperature, Power draw, ECC errors                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Alerting Rules**:
- `loss_spike > 3σ` → Auto-pause training, rollback checkpoint, notify on-call
- `gpu_utilization < 70% for 10min` → Data pipeline bottleneck alert
- `mfu < 40%` → Communication or compute efficiency issue
- `checkpoint_write_time > 30min` → Storage performance degradation

### 7.5 MLOps Pipeline

```
Code Repository (Git)
  │
  ├── Code Review → CI (lint, type-check, unit test)
  │
  ▼
Training Job Submission (CLI / Web UI)
  ├── Config validation (YAML schema)
  ├── Resource allocation (Slurm / K8s Volcano)
  ├── Environment build (Docker + uv sync)
  └── Data readiness check (manifest integrity)
  │
  ▼
Training Execution
  ├── Auto mixed precision + FSDP config
  ├── Crash-resume (CrashLoopBackOff with checkpoint)
  ├── Periodic eval (validation loss + physics consistency probe)
  └── Metrics → MLflow / W&B
  │
  ▼
Model Registry (MLflow Model Registry)
  ├── Versioned + metadata (training config, data version, metrics)
  ├── Staging: auto-deploy to test inference environment
  ├── Production: manual approval before rollout
  └── A/B testing: canary deployment
```

### 7.6 Cost Estimate

**First-Year Total Infrastructure Cost** (cloud 3-year reserved + on-demand mix):

| Category | Resources | Monthly Cost | Annual Cost |
|----------|----------|-------------|-------------|
| **Training GPU** | 32×H100 3yr reserved | ~$75,000 | ~$900,000 |
| **Annotation GPU** | 50×A100 spot + reserved | ~$35,000 | ~$420,000 |
| **Inference GPU** | 8×A100 reserved + burst | ~$12,000 | ~$144,000 |
| **CPU Nodes** | 200 vCPU (annotation+inference) | ~$8,000 | ~$96,000 |
| **Hot Storage** (NVMe) | 500 TB WekaFS | ~$25,000 | ~$300,000 |
| **Warm Storage** (SSD) | 2 PB MinIO | ~$8,000 | ~$96,000 |
| **Cold Storage** (HDD) | 10 PB object storage | ~$15,000 | ~$180,000 |
| **Network** | InfiniBand + bandwidth | ~$10,000 | ~$120,000 |
| **Personnel** | 5-8 people (incl. annotation mgmt) | ~$80,000 | ~$960,000 |
| **Other** (software licenses, etc.) | | ~$5,000 | ~$60,000 |
| **Total** | | **~$273,000/mo** | **~$3,276,000/yr** |

> Note: Above estimates are for public cloud (AWS/GCP). On-premise data centers can reduce hardware costs by 40-60%, but require additional ops personnel.

---

## 8. Team & Timeline

### 8.1 Core Team

| Role | Headcount | Responsibilities |
|------|----------|-----------------|
| **Research Lead** | 1 | Model architecture design, training strategy decisions |
| **Data Engineer** | 2 | Crawlers, data pipeline, annotation pipeline |
| **Training Engineer** | 2 | Distributed training, FSDP/DeepSpeed, stability |
| **Infra Engineer** | 2 | GPU cluster, storage, networking, K8s/Slurm |
| **ML Engineer (Post-training)** | 1 | SFT, RLHF, RM training, safety alignment |
| **ML Engineer (Inference)** | 1 | Inference optimization, cascade pipeline, online serving |
| **Annotation Manager** | 1 | Annotation quality control, crowdsourcing management |
| **Total** | **8-10** | |

### 8.2 Proposed Timeline

```
Month 1-2:  Infrastructure Setup
  ├── GPU cluster procurement/rental, network setup
  ├── Storage system deployment and stress testing
  ├── Monitoring system rollout
  └── CI/CD + MLOps pipeline setup

Month 2-4:  Data + Annotation Pipeline
  ├── Large-scale crawler development and deployment
  ├── Training data cleaning pipeline
  ├── Annotation cluster setup (130×A100 for annotation)
  └── First 10M annotations complete

Month 3-8:  Pre-Training (8 stages × ~0.75 months)
  ├── Stage 1-2: Video+Audio pretraining → baseline model
  ├── Stage 3-4: AV joint + Physics DPO → physics-consistent model
  └── Stage 5-6: High-res + long-sequence → 2.5 core capabilities

Month 7-9:  Post-Training
  ├── RM training (10K human preference annotations, 4 weeks)
  ├── SFT data construction (50K shot annotations, 6 weeks)
  ├── SFT fine-tuning (2 weeks)
  └── RLHF PPO (2 weeks)

Month 9-10: Inference Pipeline + Safety Alignment
  ├── Inference optimization (TensorRT, quantization, distillation)
  ├── Online service deployment (K8s)
  ├── Safety alignment testing
  └── Internal beta testing

Month 10-12: Polish + Release
  ├── Fix feedback issues
  ├── Performance optimization
  ├── Documentation + usage guides
  └── Public release / closed beta
```

---

## 9. Key Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|------------|
| **MoE training instability** | Medium | High | Small-scale ablation (8B MoE→30B→200B), μP tuning |
| **200B training divergence mid-run** | Medium | High | Frequent checkpoints (every 500 steps), auto-rollback |
| **Data copyright litigation** | Medium | Extreme | Legal review of training data sources, use only clearly licensed data |
| **Inference cost too high** | High | Medium | Model distillation (200B→7B student), 4-step sampling |
| **Insufficient physics consistency** | Medium | Medium | Increase PhysicsRM data, enable PhaseLock by default |
| **GPU supply shortage** | High | High | Multi-cloud strategy, advance reservation, spot complement |
| **4K output quality inadequate** | Medium | Medium | Increase Stage 5 iterations, consider dedicated SR model |
| **Team hiring difficulty** | Medium | High | Internal training, adopt mature open-source components to reduce custom needs |

---

## 10. Summary

From Seedance 2.0 (proof-of-concept open-source implementation) to Seedance 2.5 (production-grade 4K 30s commercial system), the core gap is not algorithmic approach — the current codebase already covers the major technical directions (DB-DiT, CBGA, MoE, Flow Matching, physics probes, cascade pipeline, etc.). The real gaps are:

1. **Scale**: Data (265 GB → PB-scale), compute (single node → thousand-GPU cluster), model (1.6B → 200B)
2. **Engineering**: From "it runs" to "it trains stably for months without crashing"
3. **Data quality**: From public datasets to carefully curated, multi-dimensionally annotated production data
4. **Post-training**: From none to a complete SFT + RLHF + safety alignment pipeline
5. **Infrastructure**: From single-machine scripts to a distributed training platform + inference service

Recommended approach: First run through the 30B MoE full pipeline (4×H100, 3 months), validate all engineering links, then scale to 200B.
