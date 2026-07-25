# Flux — Design Document

> **Dual-Branch Diffusion Transformer for Native Audio-Video Joint Generation**

## 1. Project Overview

Flux is a dual-branch diffusion transformer for native audio-video joint generation, developed by TopoSeek Inc. It is an open-source reference implementation inspired by the ByteDance Seedance architecture, supporting T2VA and I2VA generation.

### 1.1 Core Capabilities

| Capability | Description |
|------------|-------------|
| **T2VA** | Text description → joint video + audio generation |
| **I2VA** | Input image + text → video + audio (first-frame conditioning) |
| **Multi-resolution** | 256×256 training, cascade super-resolution up to 4K |
| **Long-duration** | 30s+ long video generation (cascaded temporal extension) |
| **Multi-scale Models** | Small (0.4B) / Base (1.6B) / 30B Dense / 30B MoE / 200B MoE |
| **RLHF** | 5-dimensional reward model + PPO reinforcement learning fine-tuning |
| **SFT** | Character consistency + facial keypoints + shot control supervised fine-tuning |

### 1.2 Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.10.0, CUDA 12.8
- **Distributed**: FSDP (FULL_SHARD), DeepSpeed ZeRO-3, torchrun
- **Attention Backends**: xformers > flash-attn > PyTorch SDPA (auto-selection)
- **Mixed Precision**: bf16 (primary), fp16 (GradScaler fallback)
- **Configuration**: OmegaConf YAML
- **Monitoring**: TensorBoard + Weights & Biases
- **Package Management**: uv (locked dependencies)

---

## 2. System Architecture

```
                    ┌──────────────────┐
                    │  T5 Text Encoder │
                    │  (google/t5-v1_1)│
                    └────────┬─────────┘
                             │ text_emb (B, L, D)
              ┌──────────────┼──────────────┐
              ▼              │               ▼
    ┌─────────────────┐      │     ┌─────────────────┐
    │   Vision Branch │◄─────┼────►│   Audio Branch  │
    │   (STDiT)       │      │     │   (DiT)         │
    │                 │      │     │                 │
    │ spatial attn    │◄─────┼────►│ self attn       │
    │ temporal attn   │ CBGA │     │ cross-text attn │
    │ cross-text attn │      │     │ FFN             │
    │ FFN / MoE       │      │     │                 │
    │                 │      │     │ 1D RoPE         │
    │ MM-RoPE (3D)    │      │     │                 │
    └────────┬────────┘      │     └────────┬────────┘
             │               │              │
             ▼               │              ▼
      VideoVAE Decoder       │       AudioVAE Decoder
      (3D CausalConv3D)      │       (2D Conv)
             │               │              │
             ▼               │              ▼
     Video Frames            │       Audio Waveform
    (B, 3, T, H, W)          │       (B, 1, T_samples)
```

### 2.1 Data Flow Overview

```
Text Input → T5 Encoder → text_emb
                            │
Noise → Flow Matching ODE ─► DB-DiT ─► Velocity Field Prediction
                            │              │
                Video Latent ←── VideoVAE Encoder
                Audio Latent ←── AudioVAE Encoder
                            │
                Decode ←── VideoVAE Decoder → Video Frames
                Decode ←── AudioVAE Decoder → Audio Waveform
```

---

## 3. Model Components

### 3.1 VideoVAE — 3D Video Autoencoder

**Files**: [flux/models/video_vae/](flux/models/video_vae/)

| Property | Value |
|----------|-------|
| Input | `(B, 3, T, H, W)` — RGB video frames |
| Output Latent | `(B, 16, T/4, H/8, W/8)` |
| Spatial Compression | 8× (2×2×2 over three stages) |
| Temporal Compression | 4× (2×2×1 over two stages) |
| Total Compression Ratio | 8 × 8 × 4 = 256× |
| Backbone | CausalConv3D ResNet |
| Regularization | KL divergence (diagonal Gaussian posterior) |
| Discriminator | 3D PatchGAN |

**Design Highlights**:
- **CausalConv3D**: Temporal convolutions are causal — frame t only depends on frames ≤ t, preventing future information leakage
- **SDXL Initialization**: Supports initializing 2D weights from SDXL VAE. 2D Conv kernels are expanded to 3D (centered at the middle temporal position); temporal layers remain zero-initialized
- **GroupNorm**: 32 groups, sharing similar spatial normalization to SDXL

**Encoder Channel Configuration** (default):
```
Stage 0: 3 → 128, stride (T:2, H:1, W:1)
Stage 1: 128 → 256, stride (T:2, H:2, W:2)
Stage 2: 256 → 512, stride (T:1, H:2, W:2)
Stage 3: 512 → 512, stride (T:1, H:2, W:2) → 2×16 ch = 32 → mean(16) + logvar(16)
```

### 3.2 AudioVAE — 2D Audio Autoencoder

**Files**: [flux/models/audio_vae/](flux/models/audio_vae/)

| Property | Value |
|----------|-------|
| Input | Mel-Spectrogram `(B, 1, 80, T_frames)` |
| Output Latent | `(B, 8, F_lat, T_lat)` |
| Sample Rate | 16 kHz |
| Mel Bands | 80 |
| Hop Length | 256 |
| Backbone | 2D Conv ResNet |
| Regularization | KL divergence |

**End-to-end Pipeline**: `waveform → MelSpectrogram → Encoder → Latent → Decoder → Mel → Griffin-Lim/inverse transform → waveform`

### 3.3 DB-DiT — Dual-Branch Diffusion Transformer

**File**: [flux/models/db_dit/db_dit.py](flux/models/db_dit/db_dit.py)

The core model, composed of:

#### 3.3.1 Overall Structure

```
Input:
  v_latent (B, 16, T, H, W)     → Video Patch Embed → v_tokens (B, N_v, D)
  a_latent (B, 8, F, T_a)       → Audio Patch Embed → a_tokens (B, N_a, D)
  t (B,)                         → Timestep Embed    → t_emb (B, D)
  text_emb (B, L_text, D_text)   → (kept as-is)

Output:
  v_pred (B, 16, T, H, W)  — video velocity field
  a_pred (B, 8, F, T_a)   — audio velocity field
```

#### 3.3.2 DualBranchBlock — Dual-Branch Transformer Layer

**File**: [flux/models/db_dit/dual_branch_block.py](flux/models/db_dit/dual_branch_block.py)

Each layer executes four sequential steps:

```
┌─────────────────────────────────────────────────────────┐
│ Layer i                                                  │
│                                                          │
│  Video Tokens (B, N_v, D)    Audio Tokens (B, N_a, D)   │
│         │                           │                    │
│  ┌──────▼──────────────────────┐ ┌──▼──────────────────┐ │
│  │ Vision Branch (STDiT)      │ │ Audio Branch (DiT)  │ │
│  │ 1. Spatial Self-Attn       │ │ 1. Self-Attn        │ │
│  │ 2. Temporal Self-Attn      │ │ 2. Cross-Text Attn  │ │
│  │ 3. Cross-Text Attn         │ │ 3. FFN / MoE        │ │
│  │ 4. FFN / MoE               │ │                      │ │
│  └──────┬──────────────────────┘ └──┬──────────────────┘ │
│         │                           │                    │
│  ┌──────▼───────────────────────────▼──────────────────┐ │
│  │ CBGA (if layer_i in cbga_layers)                   │ │
│  │  - Audio queries Video (v2a_attn + gate_a)         │ │
│  │  - Video queries Audio (a2v_attn + gate_v)         │ │
│  │  - Warmup: gates linearly 0→1 over 50K steps       │ │
│  └──────┬──────────────────────────────────────────────┘ │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────────┐ │
│  │ LipSync Bridge (if layer_i in lip_sync_layers)     │ │
│  │  - Mouth ROI Cross-Attention (video↔audio)        │ │
│  │  - Viseme embedding projection                     │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**Gradient Checkpointing**:
- Controlled via `_grad_ckpt` flag; automatically enabled for single-GPU training
- Checkpoint at the `DualBranchBlock` level — saves ~90% activation memory at ~20% extra compute
- Both branches execute independently; compatible with FSDP-level activation checkpointing

#### 3.3.3 VisionBranch — STDiT Vision Branch

**File**: [flux/models/db_dit/vision_branch.py](flux/models/db_dit/vision_branch.py)

| Sub-layer | Operation | Description |
|-----------|-----------|-------------|
| **Spatial Self-Attn** | Per-frame H×W self-attention | reshape: `(B×T, H×W, D)` |
| **Temporal Self-Attn** | Cross-frame same-position attention | reshape: `(B×H×W, T, D)` |
| **Cross-Text Attn** | Video tokens query text | standard cross-attention |
| **FFN / MoE** | Per-token feed-forward | MLP or MoE |

**Cross-Scale Attention** (optional):
- Replaces standard temporal self-attention
- Builds multi-scale pyramid `(T, T/2, T/4)` for coarse-to-fine temporal reasoning
- Uses causal masking, supporting next-frame prediction for World Model

**AdaLN Conditioning**:
- Each sub-layer has independent `AdaLN(shift, scale, gate)` modules
- Timestep t modulates all normalization parameters via t_emb
- Gates initialized to zero for stable training onset

#### 3.3.4 AudioBranch — DiT Audio Branch

**File**: [flux/models/db_dit/audio_branch.py](flux/models/db_dit/audio_branch.py)

| Sub-layer | Description |
|-----------|-------------|
| **Self-Attn** | Audio token self-attention (flattened frequency×time sequence) |
| **Cross-Text Attn** | Audio tokens query text |
| **FFN / MoE** | Per-token feed-forward |

#### 3.3.5 CBGA — Cross-Branch Gated Attention Bridge

**File**: [flux/models/db_dit/cross_modal_bridge.py](flux/models/db_dit/cross_modal_bridge.py)

This is the key innovation of Seedance 2.0 — bidirectional communication between video and audio branches:

```
Audio ──► v2a_attn(query: audio, key/value: video) ──► Audio (enhanced)
Video ──► a2v_attn(query: video, key/value: audio) ──► Video (enhanced)
```

**Gating Mechanism**:
- Each direction has a learnable scalar gate (initialized to zero)
- Gate value = `warmup_scale(t) × learnable_gate × sigmoid(t_proj(t_emb))`
- Linear warmup: 0 → 1 over 50,000 steps
- Timestep modulation: different noise levels → different modality interaction strengths

**Deployment Strategy**:
- CBGA is not at every layer; only at specific layers (e.g., `[6, 12, 18]` in a 24-layer model)
- Other layers: branches operate independently

#### 3.3.6 MM-RoPE — Multi-Modal Rotary Position Embedding

**File**: [flux/models/db_dit/mm_rope.py](flux/models/db_dit/mm_rope.py)

Partitions the head dimension into independent frequency subspaces, each encoding one positional dimension:

```
head_dim allocation:
  ┌───── temporal ─────┬──── spatial H ────┬──── spatial W ────┬── remainder ──┐
  │   rope_dim_t       │   rope_dim_h      │   rope_dim_w      │ padding       │
  └────────────────────┴───────────────────┴───────────────────┴───────────────┘
```

| Axis | Meaning | Frequency Base | Example (Base 24-layer) |
|------|---------|---------------|--------------------------|
| T (temporal) | Frame index | θ=10000 | 341 dims |
| H (height) | Row position | θ=10000 | 341 dims |
| W (width) | Column position | θ=10000 | 342 dims |
| A (audio) | 1D audio sequence position | θ=10000 | 64 dims (entire head_dim) |

**Design Rationale**: Distinct frequency bases allow the model to differentiate temporal shifts, spatial shifts, and audio shifts.

#### 3.3.7 MoE — Mixture-of-Experts Feed-Forward Network

**File**: [flux/models/db_dit/moe.py](flux/models/db_dit/moe.py)

| Property | Value |
|----------|-------|
| Routing strategy | Top-K (K=2) |
| Number of experts | 32 (configurable) |
| Shared expert | Yes (DeepSeek-style) |
| Load balancing | Switch Transformer auxiliary loss |
| Router Z-loss | DeepSeek-style log-sum-exp stabilization |

**Parameter Comparison**:
| Configuration | Total Params | Activated Params per Token |
|--------------|-------------|---------------------------|
| Standard FFN (ratio=4) | 8D² | 8D² |
| MoE-32 (ratio=1) | ~64D² | ~4D² |
| MoE-200B | 200B | ~6B (3%) |

#### 3.3.8 MultiHeadAttention — Multi-Head Attention

**File**: [flux/models/db_dit/attention.py](flux/models/db_dit/attention.py)

**Auto-selection Backend**:
1. **xformers** — Windows/Linux universal, GPU compute capability ≤9.0 (excludes Blackwell)
2. **flash-attn** — Linux optional, fp16/bf16
3. **PyTorch SDPA** — Universal fallback, natively supports attention masks

**QK Normalization**: RMSNorm on Q/K (from SD3/Flux design), configurable toggle

#### 3.3.9 Model Variant Summary

| Variant | Layers | Hidden Dim | Heads | Params | Training VRAM | Config File |
|---------|--------|-----------|-------|--------|---------------|-------------|
| Small | 12 | 768 | 12 | ~0.4B | ~25GB | [db_dit_small.yaml](../configs/model/db_dit_small.yaml) |
| Base | 24 | 1024 | 16 | ~1.6B | ~72GB | [db_dit_base.yaml](../configs/model/db_dit_base.yaml) |
| 30B Dense | 48 | 2048 | 32 | ~30B | ~160GB (8×A100) | [db_dit_30b.yaml](../configs/model/db_dit_30b.yaml) |
| 30B MoE | 48 | 2048 | 32 | ~30B | ~80GB (4×A100) | [db_dit_30b_moe.yaml](../configs/model/db_dit_30b_moe.yaml) |
| 200B MoE | 48 | 4096 | 32 | ~200B | NVMe offload | [db_dit_200b_moe.yaml](../configs/model/db_dit_200b_moe.yaml) |
| 4K 30s | 32 | 4096 | 32 | ~200B | NVMe + Seq Parallel | [db_dit_4k_30s.yaml](../configs/model/db_dit_4k_30s.yaml) |

---

## 4. Diffusion Framework — Flow Matching

**File**: [flux/diffusion/flow_matching.py](flux/diffusion/flow_matching.py)

### 4.1 Theory

Uses the **Rectified Flow** framework (Lipman et al., 2023; Esser et al., 2024):

- **Forward process**: `x_t = (1-t)·x₀_noise + t·x₁_clean`, where t ∈ [0, 1]
- **Prediction target**: velocity field `v = x₁ - x₀ = clean - noise`
- **Loss**: `MSE(v_pred, v_target)`

### 4.2 Timestep Sampling

Uses **Logit-Normal Schedule**:
- Samples t concentrated in the middle region (where noise levels change most)
- More efficient than uniform sampling

### 4.3 Sampling (Inference)

Two ODE solvers supported:

| Solver | Order | Evals per Step | Characteristics |
|--------|-------|---------------|-----------------|
| **Euler** | 1st-order | 1× | Fast, acceptable quality |
| **Heun** | 2nd-order | 2× (predict + midpoint correction) | Higher quality, default choice |

**CFG (Classifier-Free Guidance)**:
- Independent video CFG scale (default 5.0)
- Independent audio CFG scale (default 4.0)
- `v_pred = v_uncond + cfg_video × (v_cond - v_uncond)`

**I2VA First-Frame Conditioning**:
- First-frame latent replaces noise at its position
- First-frame mask forces model to predict zero velocity there

---

## 5. Training System

### 5.1 Multi-Stage Training Curriculum

| Stage | Description | Resolution | Steps | Config File |
|-------|-------------|------------|-------|-------------|
| **Stage 1** | Video pretraining (vision branch) | 256×256, 16-32fr | 500K | [stage1_video_pretrain.yaml](../configs/train/stage1_video_pretrain.yaml) |
| **Stage 2** | Audio pretraining (audio branch) | 16kHz mel | 200K | [stage2_audio_pretrain.yaml](../configs/train/stage2_audio_pretrain.yaml) |
| **Stage 3** | AV joint training (CBGA activated) | 256×256, 16-32fr | 300K | [stage3_av_joint.yaml](../configs/train/stage3_av_joint.yaml) |
| **Stage 4** | High-resolution fine-tuning | 512×512, 64fr | 100K | [stage4_hires_finetune.yaml](../configs/train/stage4_hires_finetune.yaml) |

**Initialization Strategy**:
- Stage 1: Vision branch from PixArt-α, temporal layers zero-init, audio branch frozen
- Stage 2: Audio branch trained from scratch, vision branch frozen
- Stage 3: Both branches loaded from respective stages, CBGA gates zero-init
- Stage 4: Continue from Stage 3, increase resolution and frame count

### 5.2 Trainer Design

**File**: [flux/training/trainer.py](flux/training/trainer.py)

```
Trainer
├── optimizer (AdamW, grouped weight decay)
├── scheduler (Cosine + Warmup)
├── EMA (0.9999 decay)
├── AMP Scaler (bf16/fp16)
├── FlowMatching Loss
├── WandB + TensorBoard logging
├── Checkpoint (FSDP-aware, main process only)
└── TrainingState (step, epoch, best_loss, history)
```

**Key Design Decisions**:
- **Optimizer parameter groups**: Bias and Norm params get no weight decay; all others get weight decay
- **Mixed precision**: bf16 (AMP autocast); GradScaler enabled for fp16
- **Gradient accumulation**: `effective_batch = batch_size_per_gpu × num_gpus × grad_accum_steps`
- **CBGA warmup**: Updated via `model.set_step(step)` after each optimizer step

### 5.3 Distributed Training

**File**: [flux/training/distributed.py](flux/training/distributed.py)

| Feature | Implementation |
|---------|---------------|
| Launcher | torchrun (PyTorch native) |
| Data parallelism | DistributedSampler (shuffle per epoch) |
| Model parallelism | FSDP (FULL_SHARD) with auto-wrap on DualBranchBlock |
| Mixed precision | FSDP MixedPrecision (param/reduce/buffer = bf16) |
| Gradient checkpointing | FSDP activation checkpointing on DualBranchBlock |
| Loss synchronization | all_reduce averaging across all ranks |
| Checkpointing | FSDP FULL_STATE_DICT consolidation (main process only) |

**Auto-detection Modes**:
1. **torchrun multi-node**: Uses RANK/WORLD_SIZE/LOCAL_RANK environment variables
2. **Single-node multi-GPU**: Auto-detects `torch.cuda.device_count()`, initializes NCCL
3. **Single GPU**: No distributed wrapper, manual grad ckpt

**DeepSpeed Backend**: For 30B+ MoE models and 4K 30s ultra-long sequences
- ZeRO-3 (FSDP equivalent)
- CPU/NVMe offloading (200B models)
- Ulysses sequence parallelism (4K 30s)

### 5.4 SFT — Supervised Fine-Tuning

**File**: [flux/training/sft_trainer.py](flux/training/sft_trainer.py)

Control-conditioned losses superimposed on the base Flow Matching loss:

| Loss | Weight | Description |
|------|--------|-------------|
| Flow Matching | 1.0 | Base video+audio velocity field prediction |
| LFA Consistency | 0.6 | Cross-frame character identity preservation (Identity Anchor) |
| KP Reconstruction | 0.4 | Facial 3D keypoint reconstruction |
| AV Sync | 0.1 | Audio-visual alignment contrastive loss |
| Shot Control | 0.05 | Camera type + shot scale embedding consistency |

### 5.5 RLHF — Reinforcement Learning Fine-Tuning

**Files**: [flux/training/rlhf_ppo.py](flux/training/rlhf_ppo.py) and [flux/models/reward_model.py](flux/models/reward_model.py)

**Reward Model (5 dimensions)**:
| Dimension | Weight | Scoring Head Design |
|-----------|--------|---------------------|
| visual_quality | 0.20 | Adaptive 3D pooling + MLP |
| motion_smoothness | 0.25 | Temporal difference 3D Conv + MLP |
| character_consistency | 0.25 | Cross-frame variance → consistency mapping |
| av_sync | 0.15 | Video+audio feature concat + MLP |
| prompt_alignment | 0.15 | Video-Text cosine similarity |

**PPO Training Flow**:
1. Sample K candidates from current policy (Best-of-N sampling)
2. Score with Reward Model
3. Subtract KL penalty (vs frozen SFT reference model)
4. Select best candidate or apply PPO clip update to all candidates
5. Adaptively adjust KL coefficient

### 5.6 Physics Probe

**File**: [flux/physics/physics_probe.py](flux/physics/physics_probe.py)

Based on Esmati et al. (2026)'s finding: physical plausibility is linearly decodable from DiT hidden states (81.27% accuracy).

- **PhysicsProbe**: Linear classifier trained on frozen DiT intermediate layer features
- **PhysicsProbeLoss**: Auxiliary training loss encouraging DiT to produce more physically plausible hidden representations
- 12 physics violation categories (object permanence, gravity, collision, momentum, etc.)

---

## 6. Physics Consistency Design

Physics consistency is a core challenge in video generation. Diffusion models excel at visual quality but often violate basic physical laws — objects disappearing, unnatural trajectories, missing collision detection, wrong gravity direction. Flux systematically addresses this through **six complementary mechanisms** spanning four phases: pre-training (data annotation), training (loss constraints + DPO), post-training (probe monitoring), and inference (motion prior + trajectory injection).

### 6.1 Design Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Physics Consistency Framework                        │
│                                                                      │
│  Pre-training            Training                 Inference          │
│  ┌──────────────┐   ┌──────────────┐      ┌──────────────────┐      │
│  │ VPT          │   │ World Model  │      │ PhaseLock        │      │
│  │ Role-aware   │   │ Loss         │      │ 2-step coarse→   │      │
│  │ captioning    │   │ Future pred  │      │ fine motion lock │      │
│  │ Modality-    │   │ Temporal     │      │ Latent Delta     │      │
│  │ decoupled    │   │ consistency  │      │ Guidance         │      │
│  │ noise        │   │ Physics      │      └──────────────────┘      │
│  └──────────────┘   │ plausibility │                                 │
│                     └──────────────┘      ┌──────────────────┐      │
│  ┌──────────────┐                         │ CausalMotion     │      │
│  │ Physics      │   ┌──────────────┐      │ VLM keyframe     │      │
│  │ Annotation   │   │ PhysCorr     │      │ decomposition    │      │
│  │ 12 violation │   │ PhysicsRM    │      │ Object trajectory│      │
│  │ categories   │   │ PhyDPO FT    │      │ constraints      │      │
│  └──────────────┘   └──────────────┘      │ Zero training    │      │
│                                            └──────────────────┘      │
│                     ┌──────────────┐                                │
│                     │ PhysicsProbe │  ← Post-training monitoring     │
│                     │ Linear probe │                                │
│                     │ Hidden→Score │                                │
│                     └──────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

| Mechanism | Phase | Source Paper | Core Idea |
|-----------|-------|-------------|-----------|
| **VPT** (Role-Aware) | Pre-training + Training | Zheng et al., 2026 | Tag objects with physical roles + modality-decoupled noise |
| **World Model Loss** | Training | VideoWorld 2 | Future frame prediction + temporal consistency + collision detection |
| **PhysCorr** (PhysicsRM + DPO) | Training | Wang et al., 2025 | Small reward model scoring → DPO preference optimization |
| **PhysicsProbe** | Post-training monitoring | Esmati et al., 2026 | DiT hidden states → linear decode physical plausibility |
| **PhaseLock** | Inference | Han et al., ICML 2026 | 2-step coarse denoising locks motion prior, prevents visual refinement from overwriting |
| **CausalMotion** | Inference | Zhuang et al., 2026 | VLM decomposes prompt → keyframes + trajectories, soft constraint injection |

### 6.2 VPT — Role-Aware Training + Modality-Decoupled Noise

**File**: [flux/physics/vpt.py](flux/physics/vpt.py)

#### 6.2.1 Role-Aware Captioner

Core insight: let the model understand each object's **physical role**, not just its semantic label.

```
Input:  "A person pushes a box across the floor"
Output: "[agent: person] pushes [controlled: box] across [passive: floor]"
```

| Role | Meaning | Examples |
|------|---------|---------|
| **agent** | Entity that initiates force/motion | person, robot, hand, car, dog |
| **controlled** | Entity being acted upon | ball, box, cup, door, bicycle |
| **passive** | Static environment element | floor, wall, table, ground, water |
| **background** | Scene context (no interaction) | sky, ceiling, shelf |

By annotating physical roles, the T5 encoder produces text embeddings carrying causal structure — the model implicitly learns that `agent` objects can initiate motion while `passive` objects cannot.

#### 6.2.2 Modality-Decoupled Noise

Standard Flow Matching applies the **same noise level** to all modalities, causing appearance and motion to be treated identically and blurring causal physical structure.

**VPT solution**: Visual and optical flow modalities use independent noise schedules.

```
t_visual ~ LogitNormal(0, 1)        # standard sampling
t_flow   ~ LogitNormal(0.3, 0.5)    # biased toward t=1 (cleaner), preserves motion signal
```

This ensures motion signal is always "cleaner" than visual texture, so the model prioritizes physical motion structure during denoising.

### 6.3 World Model Loss — Self-Supervised Physics Constraints

**File**: [flux/loss/world_model_loss.py](flux/loss/world_model_loss.py)

World Model Loss is a **purely visual self-supervised** loss requiring no text labels. It has three sub-losses:

#### 6.3.1 Future Frame Prediction (FuturePredictionLoss)

```
Given frames 1..T → predict frame T+1
Loss = MSE(pred, target) + 0.1 × LPIPS(pred, target)
```

#### 6.3.2 Temporal Consistency (TemporalConsistencyLoss)

Penalizes motion discontinuity. Physically plausible video should have smooth optical flow without sudden jumps:

```
diff(t)   = frame(t+1) - frame(t)       # first-order difference (velocity)
accel(t)  = diff(t+1) - diff(t)         # second-order difference (acceleration)
jerk_loss = accel².mean()               # minimize jerk (acceleration discontinuity)
```

`jerk_loss` is a core physics concept — real-world object motion has bounded jerk; sudden velocity changes imply implausible physics.

#### 6.3.3 Physics Plausibility (PhysicsPlausibilityLoss)

Optical flow heuristic-based physics violation detection:

| Detection | Method | Physical Principle |
|-----------|--------|-------------------|
| **Collision** | Local optical flow gradient spike | Collision points cause velocity direction change |
| **Momentum violation** | Acceleration discontinuity | Velocity should not change abruptly without force |
| **Penetration penalty** | Opposing flow detection in same region | Objects should not pass through each other |

```
collision_penalty = |∇_x(flow)| + |∇_y(flow)|    # flow gradient → collision
momentum_violation = accel².mean()                # acceleration → momentum violation
```

### 6.4 PhysCorr — Physics Reward Model + DPO Fine-Tuning

**File**: [flux/physics/physcorr.py](flux/physics/physcorr.py)

#### 6.4.1 PhysicsRM — Lightweight Physics Reward Model (~0.5B)

A small 3D Conv network that scores videos on physical plausibility across two dimensions:

| Dimension | What It Evaluates |
|-----------|-------------------|
| **intra_score** (intra-object stability) | Whether individual object deformation is reasonable, shape preservation |
| **inter_score** (inter-object mechanics) | Whether collisions, pushes, occlusions are correct |
| **physics_score** | `(intra + inter) / 2` |

Architecture: `3D Conv Backbone → AdaptiveAvgPool3d → 3-layer MLP → 2-dimensional scores`

#### 6.4.2 PhyDPO — DPO Training with Physics Preference

DPO (Direct Preference Optimization) trains the model to prefer physically consistent generations:

```
1. Generate N candidate videos for the same prompt
2. Score with PhysicsRM; select highest (preferred) and lowest (dispreferred)
3. DPO Loss = -log σ(β × (log_p_preferred - log_p_dispreferred))
```

For simplified training, temporal reversal serves as a "physically impossible" negative sample — reversed playback produces backwards gravity, reversed collisions, and unnatural motion.

```
Positive: video[:, :, 0:T, :, :]    # normal temporal order
Negative: video[:, :, T:0, :, :]    # temporally reversed (breaks causality)
```

### 6.5 PhysicsProbe — Physics Monitoring Probe

**File**: [flux/physics/physics_probe.py](flux/physics/physics_probe.py)

Based on Esmati et al. (2026)'s core finding: **physical plausibility is linearly decodable from DiT hidden states at 81.27% accuracy**, outperforming V-JEPA (72.1%) and VideoMAE (69.4%).

```python
# Usage: attach a linear classifier to the last 4 layers' hidden states
probe = PhysicsProbe(dim=1024, num_layers=4, num_categories=12)
score = probe(last_4_layer_hidden_states)  # (B, 12) → 12-class physics violation prediction
```

**12 Physics Violation Categories** (based on IntPhys benchmark):

| # | Category | What It Detects |
|---|----------|-----------------|
| 1 | object_permanence | Object disappears without occlusion |
| 2 | gravity_violation | Object floats or falls upward |
| 3 | collision_penetration | Objects pass through each other |
| 4 | momentum_inconsistency | Velocity changes without external force |
| 5 | shape_deformation | Rigid objects deform implausibly |
| 6 | temporal_flicker | Objects flicker between states |
| 7 | occlusion_error | Occluded objects rendered incorrectly |
| 8 | contact_mechanics | Objects don't interact on contact |
| 9 | fluid_dynamics | Liquids behave like solids or vice versa |
| 10 | lighting_inconsistency | Shadows/lights don't match the scene |
| 11 | scale_inconsistency | Objects at wrong scale relative to scene |
| 12 | camera_physics | Camera motion violates physical constraints |

**Auxiliary Training Loss**:
```python
total_loss = flow_loss + λ_probe × warmup_factor × probe_loss
```
Backpropagating the probe gradient encourages DiT to strengthen physics information in its hidden states.

### 6.6 PhaseLock — Inference-Time Motion Prior Locking

**File**: [flux/physics/phase_lock.py](flux/physics/phase_lock.py)

Han et al. (ICML 2026)'s key finding: **2-step denoising produces more physically accurate motion than 50-step denoising**. The reason: more denoising steps "over-refine" visual details, and these refinement steps simultaneously corrupt the physical motion structure established in early steps.

#### Algorithm Flow

```
PhaseLock Sampling (two passes):

Phase A — Extract Motion Prior (2-step coarse denoising):
  noise z₀ → [2-step ODE] → motion_prior (z_motion)
  ↓ Contains accurate large-scale motion structure, but visually blurry

Phase B — Guided Refinement (N-step fine denoising):
  noise z₀ → [N-step ODE + Latent Delta Guidance] → final latent
  ↓ Each step guided by cosine similarity to motion_prior
  ↓ Preserves motion structure + gains visual detail
```

#### Latent Delta Guidance

```
delta = motion_prior - current_latent              # deviation from prior
delta_low = LowPassFilter(delta)                    # keep only low-freq (motion-scale)
delta_blended = 0.8 × delta_low + 0.2 × delta      # 80% low-freq + 20% full
current = current + lock_weight(t) × delta_blended
```

**Lock Strength Schedule**: Decays from 0.5 linearly/cosine to 0 over denoising steps, ensuring early steps lock motion while later steps release for detail.

**Overhead**: ~1.06× (one extra 2-step forward pass), negligible cost.

### 6.7 CausalMotion — VLM-Guided Keyframe + Trajectory Injection

**File**: [flux/physics/causal_motion.py](flux/physics/causal_motion.py)

Zhuang et al. (2026) propose: use a VLM (Vision-Language Model) to decompose prompts into causally consistent keyframes and object trajectories, then inject as soft constraints. **Completely training-free, inference-only**.

#### Three-Component Architecture

```
Prompt → VLM Decomposition
  ├── KeyframeSchedule: keyframe positions + Gaussian windows
  └── TrajectoryConstraint: (x,y) trajectory per object

KeyframeSchedule → temporal mask (1, K, T, 1, 1)
TrajectoryConstraint → spatial mask (T, H, W)
```

#### Injection Strategy

1. **Trajectory guidance** (t < 0.5, early steps):
   ```
   latent += guidance_strength × spatial_mask × latent
   Strength decays over time: 1 → 0
   ```

2. **Keyframe constraints** (t < 0.3, very early steps):
   ```
   latent[keyframe_t] *= (1 + keyframe_weight)
   Strengthen latent signal at keyframe positions
   ```

**Heuristic fallback when VLM unavailable**: Detect motion verbs in the prompt (17 verbs: run, jump, fall, bounce, etc.), auto-generate parabolic trajectories. Replace with CogVLM2/Video-LLaVA calls in production.

### 6.8 Synergistic Design of Physics Consistency

The six mechanisms are not independent — they form complementary **defense layers**:

```
Layer 1 — Data Layer (Pre-training)
  VPT Role Annotation: model learns causal structure from input level
  Physics Annotation: 12-class physics event auto-labeling

Layer 2 — Loss Layer (Training)
  World Model Loss: self-supervised physics constraints (future+time+collision)
  VPT Decoupled Noise: training dynamics that protect motion signal

Layer 3 — Preference Layer (Training)
  PhysCorr DPO: guide model preferences with physics reward signal

Layer 4 — Monitoring Layer (Post-training)
  PhysicsProbe: continuously monitor physics information in hidden states

Layer 5 — Inference Layer (Inference-time)
  PhaseLock: lock motion prior, prevent over-refinement
  CausalMotion: VLM trajectory constraint injection
```

**Synergistic effect**:
- Training-time mechanisms (VPT + World Model + PhysCorr) let the model **learn** physics
- Inference-time mechanisms (PhaseLock + CausalMotion) prevent **forgetting** physics
- Monitoring mechanism (PhysicsProbe) verifies physics was **actually learned**

---

## 7. Data Pipeline

### 7.1 Datasets

| Dataset | Videos | Size | Description |
|---------|--------|------|-------------|
| VoxCeleb2 | 1,092,009 | 254 GB | Talking faces, 5,994 speakers, 224×224 |
| WebVid/Pexels | 2,865 / 786 | ~5 GB | General web videos + stock footage |
| HDTF | 372 | 5.8 GB | High-res talking faces |
| **Total** | **1,093,167** | **~265 GB** | |

### 7.2 Data Processing Flow

```
Raw Video → scene_detection → quality_filter → video_caption → merged manifest.csv
                │                    │                │
           PySceneDetect      BRISQUE/clarity   BLIP-2/VideoLLaMA
           (scene splitting)  (quality filter)  (auto-captioning)
```

### 7.3 Data Loaders

**VideoDataset** (`flux/data/video_dataset.py`):
- Reads CSV Manifest → OpenCV video decode → random clip → data augmentation
- Supports FPS condition embedding

**AudioDataset** (`flux/data/audio_dataset.py`):
- Read audio → resample to 16kHz → Mel spectrogram → log compression

**AVDataset** (`flux/data/av_dataset.py`):
- Joint video+audio loading, supports I2VA first-frame conditioning (30% probability)
- CFG caption dropout (10% probability)
- Error recovery: failed samples randomly replaced

**BucketSampler** (`flux/data/bucket_sampler.py`):
- Group by (resolution, num_frames, aspect_ratio) buckets to reduce padding waste

**Collate Functions** (`flux/data/collate.py`):
- Video batch: stack video + caption list
- Audio batch: stack mel + caption list
- AV batch: stack video+mel + first-frame mask

### 7.4 Data Annotation Pipeline

**Files**: `flux/data/annotation/`

- **ScenarioClassifier**: Scene type classification (indoor/outdoor/day/night/close-up/wide shot, etc.)
- **MotionQuality**: Motion quality assessment
- **PhysicsEvents**: Physical event detection

---

## 8. Inference Pipelines

### 8.1 T2VA — Text-to-Video+Audio

**File**: [flux/pipelines/pipeline_t2va.py](flux/pipelines/pipeline_t2va.py)

```
Prompt → T5 Encoder → Flow Matching ODE (30 steps Heun)
  → VideoVAE Decoder → Video Frames (T, C, H, W)
  → AudioVAE Decoder → Audio Waveform
```

### 8.2 I2VA — Image-to-Video+Audio

**File**: [flux/pipelines/pipeline_i2va.py](flux/pipelines/pipeline_i2va.py)

```
Image + Prompt → VideoVAE Encode(first frame) + T5 Encode
  → Flow Matching ODE (first-frame conditioned)
  → VideoVAE Decode + AudioVAE Decode
```

### 8.3 Cascaded — 4K 30s

**File**: [flux/pipelines/pipeline_cascade.py](flux/pipelines/pipeline_cascade.py)

```
Stage A (Coarse):   32fr × 256×256, 30 steps  — coarse structural generation
Stage B (Temporal): 32fr → 120fr, 10 steps     — temporal interpolation + diffusion refinement
Stage C (Spatial):  256 → 512 → 1024 → 4K, 10 steps each — spatial super-resolution
```

---

## 9. Loss Functions

**Files**: [flux/loss/](flux/loss/)

| Loss | File | Purpose |
|------|------|---------|
| **Flow Matching Loss** | [flow_loss.py](../flux/loss/flow_loss.py) | Velocity field MSE + optional AV sync contrastive loss |
| **VAE Loss** | [vae_loss.py](../flux/loss/vae_loss.py) | Reconstruction (L1+LPIPS) + KL + GAN (3D PatchGAN) |
| **Lip Sync Loss** | [lip_sync_loss.py](../flux/loss/lip_sync_loss.py) | Mouth-shape sync cross-attention |
| **Sync Loss** | [sync_loss.py](../flux/loss/sync_loss.py) | AV alignment contrastive loss (InfoNCE style) |
| **World Model Loss** | [world_model_loss.py](../flux/loss/world_model_loss.py) | Future frame prediction + temporal consistency |

**Flow Matching Loss Formula**:
```
video_loss = MSE(v_pred, x₁_v - x₀_v)
audio_loss = MSE(a_pred, x₁_a - x₀_a)
sync_loss  = InfoNCE(meanpool(v_pred), meanpool(a_pred))
total      = w_v × video_loss + w_a × audio_loss + w_sync × sync_loss
             + load_balance_loss + router_z_loss  (MoE)
```

---

## 10. Configuration System

**File**: [flux/utils/config.py](flux/utils/config.py) + [configs/](../configs/)

OmegaConf YAML configuration; each training stage has its own config file.

### 10.1 Model Configs

```
configs/model/
├── db_dit_small.yaml        # 0.4B params
├── db_dit_base.yaml         # 1.6B params (default)
├── db_dit_30b.yaml          # 30B dense
├── db_dit_30b_moe.yaml      # 30B MoE
├── db_dit_200b_moe.yaml     # 200B MoE
├── db_dit_4k_30s.yaml       # 4K 30s variant
├── video_vae.yaml           # VideoVAE
└── audio_vae.yaml           # AudioVAE
```

### 10.2 Training Configs

```
configs/train/
├── stage1_video_pretrain.yaml    # Video pretraining
├── stage1_30b.yaml               # 30B variant
├── stage1_200b_moe.yaml          # 200B MoE variant
├── stage1_test.yaml              # Test config
├── stage2_audio_pretrain.yaml    # Audio pretraining
├── stage3_av_joint.yaml          # AV joint training
└── stage4_hires_finetune.yaml    # High-resolution fine-tuning
```

### 10.3 Inference Configs

```
configs/inference/
├── t2va.yaml    # Text-to-video
└── i2va.yaml    # Image-to-video
```

### 10.4 Key Configuration Items

```yaml
training:
  # Model architecture
  model:
    dim: 1024
    num_layers: 24
    num_heads: 16
    ffn_ratio: 4.0
    moe_config: {num_experts: 32, top_k: 2}  # optional

  # Data
  data:
    manifest_path: "data/manifests/train.csv"
    resolution: 256
    num_frames: 32

  # Optimization
  optimizer: {type: adamw, lr: 2.0e-4, weight_decay: 0.01}
  scheduler: {type: cosine, warmup_steps: 5000}

  # Training
  max_steps: 500000
  batch_size: 8                     # per GPU
  gradient_accumulation_steps: 2    # effective batch = 8 × 8 GPU × 2 = 128
  mixed_precision: bf16

  # Distributed
  distributed:
    backend: nccl
    fsdp_sharding_strategy: FULL_SHARD
    deepspeed_preset: "30b"         # optional: "30b", "30b_moe", "200b", "4k_30s"
```

---

## 11. Additional Design Features

### 11.1 LipSync — Lip-Sync Bridge

**File**: [flux/models/mouth_roi_attention.py](flux/models/mouth_roi_attention.py)

Additional cross-modal attention beyond CBGA, focused on the mouth region:

- **MouthRegionMask**: Gaussian attention mask focused on the lower face
- **MouthROIAttention**: Video mouth-region tokens query audio tokens via cross-attention
- **Viseme Embedding**: 14-class MPEG-4 viseme embeddings (bilabial, labiodental, dental, etc.)
- **Audio→Viseme Projection**: Predict viseme classes from audio features

### 11.2 Face Analysis

**File**: [flux/models/face_analysis.py](flux/models/face_analysis.py)

Uses InsightFace/MediaPipe for:
- Face detection + landmarks
- Mouth bounding box (for precise LipSync masking)

### 11.3 LFA Encoder — Identity Feature Anchor

**File**: [flux/models/lfa_encoder.py](flux/models/lfa_encoder.py)

- Extracts global character identity features from reference frames
- Used for character consistency constraints during SFT training

### 11.4 KP Encoder — 3D Keypoint Encoder

**File**: [flux/models/kp_encoder.py](flux/models/kp_encoder.py)

- Extracts 3D projected embeddings from facial keypoints
- Serves as shot control signal

### 11.5 Cascade Pipeline — 4K 30s Generation

Enables ultra-long ultra-HD video generation via three-stage cascade, avoiding the computational explosion of one-shot generation.

---

## 12. Project Directory Structure

```
flux/
├── configs/                  # YAML configuration
│   ├── inference/            # T2VA/I2VA inference configs
│   ├── model/                # Model architecture configs
│   └── train/                # Training stage configs
├── scripts/                  # Entry point scripts
│   ├── train.py              # Training launcher (-m flux.training)
│   ├── inference_t2va.py     # T2VA inference
│   ├── inference_i2va.py     # I2VA inference
│   ├── gradio_app.py         # Web demo
│   └── build_balanced_manifest.py  # Balanced manifest builder
├── flux/                 # Main package
│   ├── models/               # Model definitions
│   │   ├── video_vae/        # VideoVAE (encoder_3d, decoder_3d, causal_conv3d, discriminator_3d, lpips_loss, resnet3d)
│   │   ├── audio_vae/        # AudioVAE (encoder, decoder, mel_transform)
│   │   ├── db_dit/           # DB-DiT (db_dit, dual_branch_block, vision_branch, audio_branch, cross_modal_bridge, mm_rope, attention, adaln, moe, qk_norm, sparse_attention, cross_scale_attention, temporal_rope)
│   │   ├── text_encoder/     # T5 encoder
│   │   ├── common/           # Shared (embedding, layers, modulation, norm)
│   │   ├── face_analysis.py  # Face analysis
│   │   ├── kp_encoder.py     # 3D keypoint encoder
│   │   ├── lfa_encoder.py    # LFA identity encoder
│   │   ├── mouth_roi_attention.py  # Lip-sync attention
│   │   └── reward_model.py   # RLHF reward model
│   ├── diffusion/            # Diffusion framework
│   │   ├── flow_matching.py  # Flow Matching training + sampling
│   │   ├── scheduler.py      # Inference scheduler
│   │   ├── guidance.py       # CFG guidance
│   │   └── noise_schedule.py # LogitNormal timestep sampling
│   ├── data/                 # Data pipeline
│   │   ├── video_dataset.py  # Video dataset
│   │   ├── audio_dataset.py  # Audio dataset
│   │   ├── av_dataset.py     # AV paired dataset
│   │   ├── bucket_sampler.py # Bucket sampler
│   │   ├── collate.py        # Batch collation
│   │   ├── transforms.py     # Data augmentation
│   │   └── annotation/       # Auto-annotation (scenario, motion, physics)
│   ├── training/             # Training system
│   │   ├── __main__.py       # Entry point
│   │   ├── trainer.py        # Base trainer
│   │   ├── sft_trainer.py    # SFT fine-tuning trainer
│   │   ├── rlhf_ppo.py       # RLHF PPO trainer
│   │   ├── distributed.py    # FSDP/DDP/DeepSpeed wrappers
│   │   ├── optimizer.py      # Optimizer factory
│   │   ├── lr_scheduler.py   # LR scheduler
│   │   └── ema.py            # EMA tracking
│   ├── loss/                 # Loss functions
│   │   ├── flow_loss.py      # Flow Matching loss
│   │   ├── vae_loss.py       # VAE reconstruction loss
│   │   ├── sync_loss.py      # AV sync contrastive loss
│   │   ├── lip_sync_loss.py  # Lip-sync loss
│   │   └── world_model_loss.py  # World Model loss
│   ├── pipelines/            # Inference pipelines
│   │   ├── pipeline_t2va.py  # T2VA pipeline
│   │   ├── pipeline_i2va.py  # I2VA pipeline
│   │   └── pipeline_cascade.py  # Cascaded 4K 30s pipeline
│   ├── physics/              # Physics probes
│   │   ├── physics_probe.py  # Physics probe + loss
│   │   ├── causal_motion.py  # Causal motion analysis
│   │   ├── phase_lock.py     # Phase locking
│   │   ├── physcorr.py       # Physics correlation
│   │   └── vpt.py            # Visual physics testing
│   ├── utils/                # Utilities
│   │   ├── config.py         # Config loading
│   │   ├── checkpoint.py     # Checkpoint save/restore
│   │   ├── logging.py        # Logging
│   │   ├── video_utils.py    # Video I/O
│   │   └── audio_utils.py    # Audio I/O
│   └── tools/                # Data prep CLI tools
│       ├── download_*.py     # Various dataset downloaders
│       ├── build_manifest.py # Manifest builder
│       ├── video_caption.py  # Auto-captioning
│       ├── scene_detection.py
│       ├── quality_filter.py
│       └── ...
├── tests/                    # Unit tests
│   ├── test_db_dit.py
│   ├── test_flow_matching.py
│   ├── test_video_vae.py
│   ├── test_mm_rope.py
│   ├── test_cbga.py
│   ├── test_annotation.py
│   ├── test_physics.py
│   └── test_pipeline.py
├── docs/                     # Documentation
│   ├── DESIGN.md             # This design document
│   ├── ROADMAP_TO_SEEDANCE_2_5.md  # Scaling roadmap
│   └── GLOBAL_DEPLOYMENT_AND_BILLING.md  # Deployment + billing
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependencies
└── README.md                 # Project overview
```

---

## 13. Testing Strategy

**Files**: [tests/](tests/)

| Test File | Coverage |
|-----------|----------|
| `test_db_dit.py` | DB-DiT forward/backward pass, output shapes |
| `test_flow_matching.py` | Flow Matching loss + sampling + CFG |
| `test_video_vae.py` | VideoVAE encode/decode, KL loss, SDXL init |
| `test_mm_rope.py` | MM-RoPE 3D + 1D rotation |
| `test_cbga.py` | CBGA gating, warmup, bidirectional attention |
| `test_annotation.py` | Auto-annotation pipeline |
| `test_physics.py` | Physics Probe training and inference |
| `test_pipeline.py` | End-to-end T2VA / I2VA pipelines |

---

## 14. Performance Benchmarks

### 14.1 Training Throughput

| Config | GPU | Batch/GPU | Effective Batch | Steps/s | Training Time (500K steps) |
|--------|-----|-----------|-----------------|---------|---------------------------|
| Small | 1× A100 40GB | 4 | 4 | ~1.2 | ~5 days |
| Base | 4× A100 80GB | 4 | 16 | ~0.8 | ~8 days |
| Base | 8× A100 80GB | 8 | 128 | ~1.0 | ~6 days |
| 30B MoE | 8× A100 80GB | 2 | 128 | ~0.3 | ~20 days |

### 14.2 Inference Speed

| Config | Frames | Resolution | Sampling Steps | Inference Time (A100) |
|--------|--------|------------|---------------|----------------------|
| Base, Euler | 32 | 256×256 | 20 | ~15s |
| Base, Heun | 32 | 256×256 | 30 | ~40s |
| 30B, Heun | 32 | 256×256 | 30 | ~120s |
| Cascade 4K | 128 | 3840×2160 | 50 | ~10min |

---

## 15. References

- [Seedance 2.0: Advancing Video Generation for World Complexity](https://arxiv.org/abs/2604.14148) — ByteDance Seed Team, 2026
- [Seedance 1.5 Pro: A Native Audio-Visual Joint Generation Foundation Model](https://arxiv.org/abs/2512.13507) — ByteDance Seed Team, 2025
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., 2023
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Esser et al., 2024
- [Open-Sora: Democratizing Efficient Video Production](https://github.com/hpcaitech/Open-Sora) — HPC-AI Tech
- [DeepSeek-V2/MoE: A Strong, Economical Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek, 2024
- [The Invisible Hand of Physics in Video Diffusion Models](https://arxiv.org/abs/2606.xxxxx) — Esmati et al., 2026
