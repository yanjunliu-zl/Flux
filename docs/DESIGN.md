# xMedia-Gen 2.0 — 设计文档

> **Seedance 2.0: Dual-Branch Diffusion Transformer for Native Audio-Video Joint Generation**

## 1. 项目概述

xMedia-Gen 2.0 是 ByteDance Seedance 2.0 架构的开源参考实现，支持文本到视频+音频（T2VA）和图像到视频+音频（I2VA）的联合生成。

### 1.1 核心能力

| 能力 | 描述 |
|------|------|
| **T2VA** | 文本描述 → 视频 + 音频联合生成 |
| **I2VA** | 输入图像 + 文本 → 视频 + 音频（首帧条件） |
| **多分辨率** | 256×256 训练，支持级联超分至 4K |
| **长时间** | 支持 30s+ 长视频生成（级联时间扩展） |
| **多模型规模** | Small (0.4B) / Base (1.6B) / 30B Dense / 30B MoE / 200B MoE |
| **RLHF** | 5 维度奖励模型 + PPO 强化学习微调 |
| **SFT** | 角色一致性 + 面部关键点 + 分镜控制的监督微调 |

### 1.2 技术栈

- **语言**: Python 3.10+
- **深度学习**: PyTorch 2.10.0, CUDA 12.8
- **分布式**: FSDP (FULL_SHARD), DeepSpeed ZeRO-3, torchrun
- **注意力后端**: xformers > flash-attn > PyTorch SDPA（自动选择）
- **混合精度**: bf16（主），fp16（GradScaler 降级）
- **配置**: OmegaConf YAML
- **监控**: TensorBoard + Weights & Biases
- **包管理**: uv（锁定依赖）

---

## 2. 总体架构

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

### 2.1 数据流概览

```
文本输入 → T5 Encoder → text_emb
                            │
噪声 → Flow Matching ODE ──► DB-DiT ──► 速度场预测
                            │              │
                    视频 Latent ←── VideoVAE Encoder
                    音频 Latent ←── AudioVAE Encoder
                            │
                    解码 ←── VideoVAE Decoder → 视频帧
                    解码 ←── AudioVAE Decoder → 音频波形
```

---

## 3. 模型组件

### 3.1 VideoVAE — 3D 视频自编码器

**文件**: [seedance/models/video_vae/](seedance/models/video_vae/)

| 属性 | 值 |
|------|---|
| 输入 | `(B, 3, T, H, W)` — RGB 视频帧 |
| 输出 Latent | `(B, 16, T/4, H/8, W/8)` |
| 空间压缩 | 8×（2×2×2 三阶段） |
| 时间压缩 | 4×（2×2×1 两阶段） |
| 总压缩比 | 8 × 8 × 4 = 256× |
| 骨干网络 | CausalConv3D ResNet |
| 正则化 | KL 散度（对角高斯后验） |
| 判别器 | 3D PatchGAN |

**设计要点**:
- **CausalConv3D**: 时间维度使用因果卷积，确保帧 t 只依赖于 ≤t 的帧，避免未来信息泄露
- **SDXL 初始化**: 支持从 SDXL VAE 的 2D 权重初始化，2D Conv 权重膨胀到 3D（中心时间位置），时间层保持零初始化
- **GroupNorm 分组**: 32 组，共享类似 SDXL 的空间归一化

**编码器通道配置**（默认）:
```
Stage 0: 3 → 128, stride (T:2, H:1, W:1)
Stage 1: 128 → 256, stride (T:2, H:2, W:2)
Stage 2: 256 → 512, stride (T:1, H:2, W:2)
Stage 3: 512 → 512, stride (T:1, H:2, W:2) → 2×16 ch = 32 → mean(16) + logvar(16)
```

### 3.2 AudioVAE — 2D 音频自编码器

**文件**: [seedance/models/audio_vae/](seedance/models/audio_vae/)

| 属性 | 值 |
|------|---|
| 输入 | Mel-Spectrogram `(B, 1, 80, T_frames)` |
| 输出 Latent | `(B, 8, F_lat, T_lat)` |
| 采样率 | 16 kHz |
| Mel 频带数 | 80 |
| Hop Length | 256 |
| 骨架 | 2D Conv ResNet |
| 正则化 | KL 散度 |

**端到端管线**: `waveform → MelSpectrogram → Encoder → Latent → Decoder → Mel → Griffin-Lim/逆变换 → waveform`

### 3.3 DB-DiT — 双分支扩散 Transformer

**文件**: [seedance/models/db_dit/db_dit.py](seedance/models/db_dit/db_dit.py)

核心模型，由以下组件构成：

#### 3.3.1 整体结构

```
输入:
  v_latent (B, 16, T, H, W)     → Video Patch Embed → v_tokens (B, N_v, D)
  a_latent (B, 8, F, T_a)       → Audio Patch Embed → a_tokens (B, N_a, D)
  t (B,)                         → Timestep Embed    → t_emb (B, D)
  text_emb (B, L_text, D_text)   → (保持原样)

输出:
  v_pred (B, 16, T, H, W)  — 视频速度场
  a_pred (B, 8, F, T_a)   — 音频速度场
```

#### 3.3.2 DualBranchBlock — 双分支 Transformer 层

**文件**: [seedance/models/db_dit/dual_branch_block.py](seedance/models/db_dit/dual_branch_block.py)

每一层按顺序执行四步：

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
│  │  - Warmup: gate 从 0 线性增长到 1 (50K steps)       │ │
│  └──────┬──────────────────────────────────────────────┘ │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────────┐ │
│  │ LipSync Bridge (if layer_i in lip_sync_layers)     │ │
│  │  - Mouth ROI Cross-Attention (video↔audio)        │ │
│  │  - Viseme embedding projection                     │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**梯度检查点（Gradient Checkpointing）**:
- 通过 `_grad_ckpt` 标志控制，单 GPU 训练时自动开启
- 在 `DualBranchBlock` 级别 checkpoint，约节省 90% 激活内存，代价约 20% 额外计算
- 双分支独立执行，支持 FSDP 级别的 activation checkpointing

#### 3.3.3 VisionBranch — STDiT 视觉分支

**文件**: [seedance/models/db_dit/vision_branch.py](seedance/models/db_dit/vision_branch.py)

| 子层 | 操作 | 描述 |
|------|------|------|
| **Spatial Self-Attn** | 每帧内 H×W 位置自注意力 | reshape: `(B×T, H×W, D)` |
| **Temporal Self-Attn** | 跨帧同空间位置注意力 | reshape: `(B×H×W, T, D)` |
| **Cross-Text Attn** | 视频 token 查询文本 | 标准 cross-attention |
| **FFN / MoE** | 逐 token 前馈 | MLP 或 MoE |

**交叉尺度注意力**（Cross-Scale Attention, 可选）:
- 替代标准 Temporal Self-Attention
- 构建多尺度金字塔 `(T, T/2, T/4)` 用于粗到细的时间推理
- 使用因果掩码，支持 World Model 的下一帧预测

**AdaLN 条件注入**:
- 每个子层都有独立的 `AdaLN(shift, scale, gate)` 模块
- 时间步 t 通过 t_emb 调制所有归一化参数
- Gate 初始化为零，确保训练初期的稳定性

#### 3.3.4 AudioBranch — DiT 音频分支

**文件**: [seedance/models/db_dit/audio_branch.py](seedance/models/db_dit/audio_branch.py)

| 子层 | 描述 |
|------|------|
| **Self-Attn** | 音频 token 自注意力（频率×时间扁平序列） |
| **Cross-Text Attn** | 音频 token 查询文本 |
| **FFN / MoE** | 逐 token 前馈 |

#### 3.3.5 CBGA — 跨模态门控注意力桥

**文件**: [seedance/models/db_dit/cross_modal_bridge.py](seedance/models/db_dit/cross_modal_bridge.py)

这是 Seedance 2.0 的关键创新——在视频和音频分支之间建立双向通信：

```
Audio ──► v2a_attn(query: audio, key/value: video) ──► Audio (增强)
Video ──► a2v_attn(query: video, key/value: audio) ──► Video (增强)
```

**门控机制**:
- 每个方向有一个可学习的标量门（初始化为 0）
- 门控值 = `warmup_scale(t) × learnable_gate × sigmoid(t_proj(t_emb))`
- 线性 warmup：0 → 1，50,000 步
- 时间步调制：不同噪声水平对应不同的模态交互强度

**部署策略**:
- 并非每层都有 CBGA，仅在特定层（如 `[6, 12, 18]` 的 24 层模型）部署
- 其他层两个分支独立运行

#### 3.3.6 MM-RoPE — 多模态旋转位置编码

**文件**: [seedance/models/db_dit/mm_rope.py](seedance/models/db_dit/mm_rope.py)

将 head 维度划分为独立的频率子空间，每个子空间编码一个位置维度：

```
head_dim 分配:
  ┌───── temporal ─────┬──── spatial H ────┬──── spatial W ────┬── 剩余 ──┐
  │   rope_dim_t       │   rope_dim_h      │   rope_dim_w      │ padding  │
  └────────────────────┴───────────────────┴───────────────────┴──────────┘
```

| 轴 | 含义 | 频率基准 | 示例 (Base 24层) |
|----|------|---------|----------------|
| T (时间) | 帧索引 | θ=10000 | 341 dims (1024/16 heads ≈ 64 per-head, 约 21 维度) |
| H (高度) | 行位置 | θ=10000 | 341 dims |
| W (宽度) | 列位置 | θ=10000 | 342 dims |
| A (音频) | 1D 音频序列位置 | θ=10000 | 64 dims (整个 head_dim) |

**设计原理**: 不同的频率基使模型能区分时间偏移、空间偏移和音频偏移。

#### 3.3.7 MoE — 混合专家前馈网络

**文件**: [seedance/models/db_dit/moe.py](seedance/models/db_dit/moe.py)

| 属性 | 值 |
|------|---|
| 路由策略 | Top-K（K=2） |
| 专家数 | 32（可配置） |
| 共享专家 | 是（DeepSeek 风格） |
| 负载均衡 | Switch Transformer 辅助损失 |
| Router Z-loss | DeepSeek 风格的 log-sum-exp 稳定化 |

**参数对比**:
| 配置 | 总参数 | 每 token 激活参数 |
|------|-------|------------------|
| 标准 FFN (ratio=4) | 8D² | 8D² |
| MoE-32 (ratio=1) | ~64D² | ~4D² |
| MoE-200B | 200B | ~6B (3%) |

#### 3.3.8 MultiHeadAttention — 多头注意力

**文件**: [seedance/models/db_dit/attention.py](seedance/models/db_dit/attention.py)

**后端自动选择**:
1. **xformers** — Windows/Linux 通用，GPU 计算能力 ≤9.0（Blackwell 除外）
2. **flash-attn** — Linux 可选，fp16/bf16
3. **PyTorch SDPA** — 通用降级方案，原生支持注意力掩码

**QK 归一化**: RMSNorm on Q/K（来自 SD3/Flux 设计），可选开关

#### 3.3.9 模型变体汇总

| 变体 | 层数 | Hidden Dim | 头数 | 参数量 | 训练 VRAM | 配置文件 |
|------|------|-----------|------|--------|----------|---------|
| Small | 12 | 768 | 12 | ~0.4B | ~25GB | [db_dit_small.yaml](../configs/model/db_dit_small.yaml) |
| Base | 24 | 1024 | 16 | ~1.6B | ~72GB | [db_dit_base.yaml](../configs/model/db_dit_base.yaml) |
| 30B Dense | 48 | 2048 | 32 | ~30B | ~160GB (8×A100) | [db_dit_30b.yaml](../configs/model/db_dit_30b.yaml) |
| 30B MoE | 48 | 2048 | 32 | ~30B | ~80GB (4×A100) | [db_dit_30b_moe.yaml](../configs/model/db_dit_30b_moe.yaml) |
| 200B MoE | 72 | 3072 | 48 | ~200B | NVMe offload | [db_dit_200b_moe.yaml](../configs/model/db_dit_200b_moe.yaml) |
| 4K 30s | 72 | 3072 | 48 | ~200B | NVMe + Seq Parallel | [db_dit_4k_30s.yaml](../configs/model/db_dit_4k_30s.yaml) |

---

## 4. 扩散框架 — Flow Matching

**文件**: [seedance/diffusion/flow_matching.py](seedance/diffusion/flow_matching.py)

### 4.1 理论

采用 **Rectified Flow** 框架（Lipman et al., 2023; Esser et al., 2024）：

- **前向过程**: `x_t = (1-t)·x₀_noise + t·x₁_clean`, 其中 t ∈ [0, 1]
- **预测目标**: 速度场 `v = x₁ - x₀ = clean - noise`
- **Loss**: `MSE(v_pred, v_target)`

### 4.2 时间步采样

使用 **Logit-Normal Schedule**（logit-normal timestep sampling）：
- 采样 t 集中在中间区域（噪声水平变化最大的地方）
- 比均匀采样更高效的训练

### 4.3 采样（推理）

支持两种 ODE 求解器：

| 求解器 | 阶数 | 每步评估次数 | 特点 |
|--------|------|-------------|------|
| **Euler** | 1 阶 | 1× | 快速，质量可接受 |
| **Heun** | 2 阶 | 2× (预测+中点校正) | 更高质量，默认选择 |

**CFG (Classifier-Free Guidance)**:
- 独立视频 CFG 尺度（默认 5.0）
- 独立音频 CFG 尺度（默认 4.0）
- `v_pred = v_uncond + cfg_video × (v_cond - v_uncond)`

**I2VA 首帧条件**:
- 首帧 latent 替换噪声的对应位置
- 首帧 mask 强制模型预测 0 速度

---

## 5. 训练系统

### 5.1 多阶段训练课程

| 阶段 | 描述 | 分辨率 | 步数 | 配置文件 |
|------|------|--------|------|---------|
| **Stage 1** | 视频预训练（视觉分支） | 256×256, 16-32fr | 500K | [stage1_video_pretrain.yaml](../configs/train/stage1_video_pretrain.yaml) |
| **Stage 2** | 音频预训练（音频分支） | 16kHz mel | 200K | [stage2_audio_pretrain.yaml](../configs/train/stage2_audio_pretrain.yaml) |
| **Stage 3** | 视听联合训练（CBGA 激活） | 256×256, 16-32fr | 300K | [stage3_av_joint.yaml](../configs/train/stage3_av_joint.yaml) |
| **Stage 4** | 高分辨率微调 | 512×512, 64fr | 100K | [stage4_hires_finetune.yaml](../configs/train/stage4_hires_finetune.yaml) |

**初始化策略**:
- Stage 1: 视觉分支从 PixArt-α 初始化，时间层零初始化，音频分支冻结
- Stage 2: 音频分支从头训练，视觉分支冻结
- Stage 3: 两个分支从各自阶段加载，CBGA 门控零初始化
- Stage 4: 从 Stage 3 继续，提高分辨率和帧数

### 5.2 Trainer 设计

**文件**: [seedance/training/trainer.py](seedance/training/trainer.py)

```
Trainer
├── optimizer (AdamW, 分组 weight decay)
├── scheduler (Cosine + Warmup)
├── EMA (0.9999 衰减)
├── AMP Scaler (bf16/fp16)
├── FlowMatching Loss
├── WandB + TensorBoard 日志
├── Checkpoint (FSDP 感知，仅主进程保存)
└── TrainingState (step, epoch, best_loss, history)
```

**关键设计**:
- **优化器分组**: Bias 和 Norm 参数无 weight decay，其他参数有 weight decay
- **混合精度**: bf16（AMP autocast），fp16 时启用 GradScaler
- **梯度累积**: `effective_batch = batch_size_per_gpu × num_gpus × grad_accum_steps`
- **CBGA Warmup**: 通过 `model.set_step(step)` 在每个优化步骤后更新

### 5.3 分布式训练

**文件**: [seedance/training/distributed.py](seedance/training/distributed.py)

| 特性 | 实现 |
|------|------|
| 启动器 | torchrun（PyTorch 原生） |
| 数据并行 | DistributedSampler (shuffle per epoch) |
| 模型并行 | FSDP (FULL_SHARD) with auto-wrap on DualBranchBlock |
| 混合精度 | FSDP MixedPrecision (param/reduce/buffer = bf16) |
| 梯度检查点 | FSDP activation checkpointing on DualBranchBlock |
| 损失同步 | all_reduce averaging across all ranks |
| 检查点 | FSDP FULL_STATE_DICT consolidation (main process only) |

**自动检测模式**:
1. **torchrun 多节点**: 使用 RANK/WORLD_SIZE/LOCAL_RANK 环境变量
2. **单机多 GPU**: 自动检测 `torch.cuda.device_count()`，初始化 NCCL
3. **单 GPU**: 无分布式包装，手动 grad ckpt

**DeepSpeed 后端**: 用于 30B+ MoE 模型和 4K 30s 超长序列
- ZeRO-3（FSDP 等效）
- CPU/NVMe 卸载（200B 模型）
- Ulysses 序列并行（4K 30s）

### 5.4 SFT 监督微调

**文件**: [seedance/training/sft_trainer.py](seedance/training/sft_trainer.py)

在基础 Flow Matching 损失之上叠加控制信号损失：

| 损失 | 权重 | 描述 |
|------|------|------|
| Flow Matching | 1.0 | 基础视频+音频速度场预测 |
| LFA 一致性 | 0.6 | 角色跨帧身份保持（Identity Anchor） |
| KP 重建 | 0.4 | 面部 3D 关键点重建 |
| AV 同步 | 0.1 | 音视频对齐对比损失 |
| 分镜控制 | 0.05 | 镜头类型+景别嵌入一致性 |

### 5.5 RLHF 强化学习微调

**文件**: [seedance/training/rlhf_ppo.py](seedance/training/rlhf_ppo.py) 和 [seedance/models/reward_model.py](seedance/models/reward_model.py)

**奖励模型（5 维度）**:
| 维度 | 权重 | 评分头设计 |
|------|------|-----------|
| visual_quality | 0.20 | 自适应 3D 池化 + MLP |
| motion_smoothness | 0.25 | 时间差分 3D Conv + MLP |
| character_consistency | 0.25 | 跨帧方差 → 一致性映射 |
| av_sync | 0.15 | 视频+音频特征拼接 + MLP |
| prompt_alignment | 0.15 | Video-Text 余弦相似度 |

**PPO 训练流程**:
1. 从当前策略采样 K 个候选（Best-of-N sampling）
2. Reward Model 评分
3. 减去 KL 惩罚（vs 冻结的 SFT 参考模型）
4. 选择最优候选或对所有候选做 PPO clip 更新
5. 自适应调整 KL 系数

### 5.6 Physics Probe — 物理探针

**文件**: [seedance/physics/physics_probe.py](seedance/physics/physics_probe.py)

基于 Esmati et al. (2026) 的发现：物理合理性可以从 DiT 隐藏状态中线性解码（81.27% 准确率）。

- **PhysicsProbe**: 线性分类器，在冻结的 DiT 中间层特征上训练
- **PhysicsProbeLoss**: 辅助训练损失，鼓励 DiT 产生物理更合理的隐藏表示
- 12 个物理违规类别（对象持久性、重力、碰撞、动量等）

---

## 6. 物理一致性设计

物理一致性是视频生成的核心挑战之一。扩散模型虽然在视觉质量上表现出色，但常常违反基本的物理规律——物体突然消失、运动轨迹不自然、碰撞检测缺失、重力方向错误等。xMedia-Gen 2.0 通过**六个互补机制**系统性地解决这个问题，覆盖了训练前（数据标注）、训练中（损失约束 + DPO）、训练后（探针监测）和推理时（运动先验 + 轨迹注入）四个阶段。

### 6.1 设计总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                    物理一致性保障体系                                  │
│                                                                      │
│  训练前                训练中                  推理时                 │
│  ┌──────────────┐   ┌──────────────┐      ┌──────────────────┐      │
│  │ VPT          │   │ World Model  │      │ PhaseLock        │      │
│  │ 角色感知标注  │   │ Loss         │      │ 2步粗→细运动锁定  │      │
│  │ 模态解耦噪声  │   │ 未来帧预测   │      │ Latent Delta     │      │
│  └──────────────┘   │ 时间一致性   │      │ Guidance         │      │
│                     │ 物理合理性   │      └──────────────────┘      │
│  ┌──────────────┐   └──────────────┘      ┌──────────────────┐      │
│  │ Physics      │                         │ CausalMotion     │      │
│  │ Annotation   │   ┌──────────────┐      │ VLM关键帧分解    │      │
│  │ 12类违规检测 │   │ PhysCorr     │      │ 对象轨迹约束     │      │
│  └──────────────┘   │ PhysicsRM    │      │ 零训练开销       │      │
│                     │ PhyDPO 微调  │      └──────────────────┘      │
│                     └──────────────┘                                │
│                     ┌──────────────┐                                │
│                     │ PhysicsProbe │  ← 训练后监测                   │
│                     │ 线性探针      │                                │
│                     │ 隐藏→物理评分 │                                │
│                     └──────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

| 机制 | 阶段 | 来源论文 | 核心思想 |
|------|------|---------|---------|
| **VPT** (Role-Aware) | 训练前+训练中 | Zheng et al., 2026 | 给物体打物理角色标签 + 模态解耦噪声 |
| **World Model Loss** | 训练中 | VideoWorld 2 | 未来帧预测 + 时间一致性 + 碰撞检测 |
| **PhysCorr** (PhysicsRM + DPO) | 训练中 | Wang et al., 2025 | 小奖励模型评分 → DPO 偏好优化 |
| **PhysicsProbe** | 训练后监测 | Esmati et al., 2026 | DiT 隐藏状态 → 线性解码物理合理性 |
| **PhaseLock** | 推理时 | Han et al., ICML 2026 | 2步粗去噪锁定运动先验，防止视觉精炼覆盖 |
| **CausalMotion** | 推理时 | Zhuang et al., 2026 | VLM 分解提示→关键帧+轨迹，软约束注入 |

### 6.2 VPT — 角色感知训练 + 模态解耦噪声

**文件**: [seedance/physics/vpt.py](seedance/physics/vpt.py)

#### 6.2.1 角色感知标注 (Role-Aware Captioner)

核心思想：让模型理解场景中每个物体的**物理角色**，而不仅仅是语义标签。

```
输入: "A person pushes a box across the floor"
输出: "[agent: person] pushes [controlled: box] across [passive: floor]"
```

| 角色 | 含义 | 示例 |
|------|------|------|
| **agent** | 主动施加力/运动的实体 | person, robot, hand, car, dog |
| **controlled** | 被作用的实体 | ball, box, cup, door, bicycle |
| **passive** | 静态环境元素 | floor, wall, table, ground, water |
| **background** | 场景上下文（无交互） | sky, ceiling, shelf |

通过标注物理角色，T5 编码器产出的 text_emb 携带了因果结构信息——模型隐式学习到 `agent` 物体可以发起运动而 `passive` 物体不能。

#### 6.2.2 模态解耦噪声 (Modality-Decoupled Noise)

标准 Flow Matching 对所有模态施加相同噪声水平，导致外观和运动被等同对待，模糊了因果物理结构。

**VPT 方案**：视觉和光流模态使用独立的噪声调度。

```
t_visual ~ LogitNormal(0, 1)        # 标准采样
t_flow   ~ LogitNormal(0.3, 0.5)    # 偏向 t=1（更干净），保留运动信号
```

这使得运动信号始终比视觉纹理"更清晰"，模型在去噪时优先保持物理运动结构。

### 6.3 World Model Loss — 自监督物理约束

**文件**: [seedance/loss/world_model_loss.py](seedance/loss/world_model_loss.py)

World Model Loss 是一种**纯视觉自监督**损失，不需要文本标注。它包含三个子损失：

#### 6.3.1 未来帧预测 (FuturePredictionLoss)

```
给定帧 1..T → 预测帧 T+1
Loss = MSE(pred, target) + 0.1 × LPIPS(pred, target)
```

#### 6.3.2 时间一致性 (TemporalConsistencyLoss)

惩罚运动的不连续性。物理合理的视频应具有平滑的光流，没有突然的跳跃：

```
diff(t)   = frame(t+1) - frame(t)       # 一阶差分（速度）
accel(t)  = diff(t+1) - diff(t)         # 二阶差分（加速度）
jerk_loss = accel².mean()               # 最小化 jerk（加速度突变）
```

`jerk_loss` 是物理学的核心概念——真实世界的物体运动具有有限的 jerk（加加速度），突然的速度变化意味着不合理的物理。

#### 6.3.3 物理合理性 (PhysicsPlausibilityLoss)

基于光流启发式的物理违规检测：

| 检测项 | 方法 | 物理原理 |
|--------|------|---------|
| **碰撞检测** | 局部光流梯度突变 | 碰撞点产生速度方向改变 |
| **动量违规** | 加速度不连续性 | 无外力时速度不应突变 |
| **穿透惩罚** | 同区域对流检测 | 物体不应相互穿过 |

```
collision_penalty = |∇_x(flow)| + |∇_y(flow)|    # 流梯度 → 碰撞
momentum_violation = accel².mean()                # 加速度 → 动量违背
```

### 6.4 PhysCorr — 物理奖励模型 + DPO 微调

**文件**: [seedance/physics/physcorr.py](seedance/physics/physcorr.py)

#### 6.4.1 PhysicsRM — 轻量物理奖励模型 (~0.5B)

一个小型 3D Conv 网络，评分视频的物理合理性，分两个维度：

| 维度 | 评估内容 |
|------|---------|
| **intra_score**（对象内稳定性） | 单个物体形变是否合理、形态是否保持 |
| **inter_score**（对象间力学） | 碰撞、推动、遮挡等交互是否正确 |
| **physics_score** | `(intra + inter) / 2` |

架构：`3D Conv Backbone → AdaptiveAvgPool3d → 3层 MLP → 2维分数`

#### 6.4.2 PhyDPO — 基于物理偏好的 DPO 训练

DPO (Direct Preference Optimization) 让模型学会偏好物理一致的生成结果：

```
1. 对同一 prompt 生成 N 个候选视频
2. PhysicsRM 评分，选出最高分（preferred）和最低分（dispreferred）
3. DPO Loss = -log σ(β × (log_p_preferred - log_p_dispreferred))
```

简化训练时，用时间翻转作为"物理上不可能"的负样本——时间倒放会产生反向重力、反向碰撞等不自然的运动。

```
正样本: video[:, :, 0:T, :, :]    # 正常时间顺序
负样本: video[:, :, T:0, :, :]    # 时间翻转（打破因果）
```

### 6.5 PhysicsProbe — 物理探针监测

**文件**: [seedance/physics/physics_probe.py](seedance/physics/physics_probe.py)

基于 Esmati et al. (2026) 的核心发现：**物理合理性可以从 DiT 隐藏状态中线性解码，准确率 81.27%**，优于 V-JEPA (72.1%) 和 VideoMAE (69.4%)。

```python
# 用法：在 DiT 最后 4 层的隐藏状态上接一个线性分类器
probe = PhysicsProbe(dim=1024, num_layers=4, num_categories=12)
score = probe(last_4_layer_hidden_states)  # (B, 12) → 12类物理违规预测
```

**12 类物理违规检测**（基于 IntPhys 基准）：

| # | 类别 | 具体检测 |
|---|------|---------|
| 1 | object_permanence | 物体在没有遮挡的情况下消失 |
| 2 | gravity_violation | 物体漂浮或向上掉落 |
| 3 | collision_penetration | 物体相互穿过 |
| 4 | momentum_inconsistency | 没有外力时速度改变 |
| 5 | shape_deformation | 刚体不合理形变 |
| 6 | temporal_flicker | 物体在状态间闪烁 |
| 7 | occlusion_error | 遮挡物体渲染错误 |
| 8 | contact_mechanics | 接触时物体不产生相互作用 |
| 9 | fluid_dynamics | 液体像固体或反之 |
| 10 | lighting_inconsistency | 阴影/灯光不匹配场景 |
| 11 | scale_inconsistency | 物体与场景比例不正确 |
| 12 | camera_physics | 相机运动违反物理约束 |

**辅助训练损失**:
```python
total_loss = flow_loss + λ_probe × warmup_factor × probe_loss
```
通过反向传播探针的梯度，鼓励 DiT 在隐藏状态中强化物理信息。

### 6.6 PhaseLock — 推理时运动先验锁定

**文件**: [seedance/physics/phase_lock.py](seedance/physics/phase_lock.py)

Han et al. (ICML 2026) 的关键发现：**2 步去噪产生的运动比 50 步去噪更物理准确**。原因在于，更多的去噪步骤会"过度精炼"视觉细节，但这些精炼步骤同时会破坏早期步骤已确立的物理运动结构。

#### 算法流程

```
PhaseLock 采样（两步走）：

阶段 A — 提取运动先验 (2步粗去噪):
  噪声 z₀ → [2步 ODE] → motion_prior (z_motion)
  ↓ 包含准确的大尺度运动结构，但视觉模糊

阶段 B — 引导精炼 (N步细去噪):
  噪声 z₀ → [N步 ODE + Latent Delta Guidance] → 最终 latent
  ↓ 每一步用 motion_prior 的余弦相似度引导
  ↓ 保留运动结构 + 获取视觉细节
```

#### Latent Delta Guidance

```
delta = motion_prior - current_latent        # 从先验的偏差
delta_low = LowPassFilter(delta)              # 只保留低频分量（运动尺度）
delta_blended = 0.8 × delta_low + 0.2 × delta # 80%低频 + 20%全频
current = current + lock_weight(t) × delta_blended
```

**锁强度调度**: 随去噪步数从 0.5 线性/余弦衰减到 0，确保早期步锁定运动，晚期步释放以增加细节。

**额外开销**: ~1.06×（一次额外的 2 步前向传播），几乎无代价。

### 6.7 CausalMotion — VLM 引导的关键帧+轨迹注入

**文件**: [seedance/physics/causal_motion.py](seedance/physics/causal_motion.py)

Zhuang et al. (2026) 提出：用 VLM（视觉语言模型）将提示分解为因果一致的关键帧和对象轨迹，然后作为软约束注入。**完全零训练，仅推理时使用**。

#### 三段式架构

```
Prompt → VLM 分解
  ├── KeyframeSchedule: 关键帧位置 + 高斯窗口
  └── TrajectoryConstraint: 每个物体的 (x,y) 轨迹

KeyframeSchedule → 时间掩码 (1, K, T, 1, 1)
TrajectoryConstraint → 空间掩码 (T, H, W)
```

#### 注入策略

1. **轨迹引导** (t < 0.5，早期步):
   ```
   latent += guidance_strength × spatial_mask × latent
   强度随时间衰减: 1 → 0
   ```

2. **关键帧约束** (t < 0.3，极早期步):
   ```
   latent[keyframe_t] *= (1 + keyframe_weight)
   在关键帧位置加强 latent 信号
   ```

**VLM 未就绪时的启发式方案**: 检测提示中的运动动词（run, jump, fall, bounce 等 17 个），自动生成抛物线轨迹。实际部署时替换为 CogVLM2/Video-LLaVA 调用。

### 6.8 物理一致性的协同设计

六个机制并非独立工作，而是形成互补的**防御层次**：

```
Layer 1 — 数据层 (训练前)
  VPT 角色标注: 让模型从输入层面理解因果结构
  Physics Annotation: 12类物理事件自动标注

Layer 2 — 损失层 (训练中)
  World Model Loss: 自监督的物理约束（未来帧+时间+碰撞）
  VPT 解耦噪声: 保护运动信号的训练动态

Layer 3 — 偏好层 (训练中)
  PhysCorr DPO: 用物理奖励信号引导模型偏好

Layer 4 — 监测层 (训练后)
  PhysicsProbe: 持续监测隐藏状态中的物理信息

Layer 5 — 推理层 (推理时)
  PhaseLock: 锁定运动先验，防止过度精炼
  CausalMotion: VLM 轨迹约束注入
```

**协同效果**：
- 训练时机制（VPT + World Model + PhysCorr）让模型**学会**物理
- 推理时机制（PhaseLock + CausalMotion）防止**遗忘**物理
- 监测机制（PhysicsProbe）验证物理是否**真的学到了**

---

## 7. 数据管线

### 6.1 数据集

| 数据集 | 视频数 | 大小 | 描述 |
|--------|--------|------|------|
| VoxCeleb2 | 1,092,009 | 254 GB | 说话人脸, 5,994 说话人, 224×224 |
| WebVid/Pexels | 2,865/786 | ~5 GB | 通用网络视频 + 库存视频 |
| HDTF | 372 | 5.8 GB | 高清说话人脸 |
| **总计** | **1,093,167** | **~265 GB** | |

### 6.2 数据处理流程

```
原始视频 → scene_detection → quality_filter → video_caption → 合并 manifest.csv
                │                    │                │
           PySceneDetect      BRISQUE/clarity   BLIP-2/VideoLLaMA
           (场景分割)          (质量过滤)        (自动标注)
```

### 6.3 数据加载器

**VideoDataset** (`seedance/data/video_dataset.py`):
- 读取 CSV Manifest → OpenCV 视频解码 → 随机剪辑 → 数据增强
- 支持 FPS 条件嵌入

**AudioDataset** (`seedance/data/audio_dataset.py`):
- 读取音频 → 重采样到 16kHz → Mel 谱图 → log 压缩

**AVDataset** (`seedance/data/av_dataset.py`):
- 联合加载视频+音频，支持 I2VA 首帧条件（30% 概率）
- CFG caption dropout（10% 概率）
- 错误恢复：失败样本随机替换

**BucketSampler** (`seedance/data/bucket_sampler.py`):
- 按 (分辨率, 帧数, 宽高比) 分桶，减少 padding 浪费

**Collate Functions** (`seedance/data/collate.py`):
- 视频批次: 堆叠视频 + 标题列表
- 音频批次: 堆叠 mel + 标题列表
- AV 批次: 堆叠视频+mel + 首帧 mask

### 6.4 数据标注管线

**文件**: `seedance/data/annotation/`

- **ScenarioClassifier**: 场景类型分类（室内/室外/白天/夜晚/特写/远景等）
- **MotionQuality**: 运动质量评估
- **PhysicsEvents**: 物理事件检测

---

## 8. 推理管线

### 7.1 T2VA — 文本到视频+音频

**文件**: [seedance/pipelines/pipeline_t2va.py](seedance/pipelines/pipeline_t2va.py)

```
Prompt → T5 Encoder → Flow Matching ODE (30 steps Heun)
  → VideoVAE Decoder → 视频帧 (T, C, H, W)
  → AudioVAE Decoder → 音频波形
```

### 7.2 I2VA — 图像到视频+音频

**文件**: [seedance/pipelines/pipeline_i2va.py](seedance/pipelines/pipeline_i2va.py)

```
Image + Prompt → VideoVAE Encode(首帧) + T5 Encode
  → Flow Matching ODE (首帧条件)
  → VideoVAE Decode + AudioVAE Decode
```

### 7.3 Cascaded — 级联 4K 30s

**文件**: [seedance/pipelines/pipeline_cascade.py](seedance/pipelines/pipeline_cascade.py)

```
Stage A (Coarse):   32fr × 256×256,  30 steps  — 粗结构生成
Stage B (Temporal): 32fr → 120fr,    10 steps  — 时间插值+扩散精炼
Stage C (Spatial):  256 → 512 → 1024 → 4K, 各 10 steps — 空间超分
```

---

## 9. 损失函数

**文件**: [seedance/loss/](seedance/loss/)

| 损失 | 文件 | 用途 |
|------|------|------|
| **Flow Matching Loss** | [flow_loss.py](../seedance/loss/flow_loss.py) | 速度场 MSE + 可选 AV sync 对比损失 |
| **VAE Loss** | [vae_loss.py](../seedance/loss/vae_loss.py) | 重建 (L1+LPIPS) + KL + GAN (3D PatchGAN) |
| **Lip Sync Loss** | [lip_sync_loss.py](../seedance/loss/lip_sync_loss.py) | 口型同步交叉注意力 |
| **Sync Loss** | [sync_loss.py](../seedance/loss/sync_loss.py) | AV 对齐对比损失 (InfoNCE style) |
| **World Model Loss** | [world_model_loss.py](../seedance/loss/world_model_loss.py) | 未来帧预测 + 时间一致性 |

**Flow Matching Loss 公式**:
```
video_loss = MSE(v_pred, x₁_v - x₀_v)
audio_loss = MSE(a_pred, x₁_a - x₀_a)
sync_loss  = InfoNCE(meanpool(v_pred), meanpool(a_pred))
total      = w_v × video_loss + w_a × audio_loss + w_sync × sync_loss
             + load_balance_loss + router_z_loss  (MoE)
```

---

## 10. 配置系统

**文件**: [seedance/utils/config.py](seedance/utils/config.py) + [configs/](../configs/)

使用 OmegaConf YAML 配置，每个训练阶段独立配置文件。

### 9.1 模型配置

```
configs/model/
├── db_dit_small.yaml       # 0.4B 参数
├── db_dit_base.yaml        # 1.6B 参数（默认）
├── db_dit_30b.yaml         # 30B 密集
├── db_dit_30b_moe.yaml     # 30B MoE
├── db_dit_200b_moe.yaml    # 200B MoE
├── db_dit_4k_30s.yaml      # 4K 30s 变体
├── video_vae.yaml          # VideoVAE
└── audio_vae.yaml          # AudioVAE
```

### 9.2 训练配置

```
configs/train/
├── stage1_video_pretrain.yaml    # 视频预训练
├── stage1_30b.yaml               # 30B 变体
├── stage1_200b_moe.yaml          # 200B MoE 变体
├── stage1_test.yaml              # 测试配置
├── stage2_audio_pretrain.yaml    # 音频预训练
├── stage3_av_joint.yaml          # AV 联合训练
└── stage4_hires_finetune.yaml    # 高分辨率微调
```

### 9.3 推理配置

```
configs/inference/
├── t2va.yaml    # 文本到视频
└── i2va.yaml    # 图像到视频
```

### 9.4 关键配置项

```yaml
training:
  # 模型架构
  model:
    dim: 1024
    num_layers: 24
    num_heads: 16
    ffn_ratio: 4.0
    moe_config: {num_experts: 32, top_k: 2}  # 可选

  # 数据
  data:
    manifest_path: "data/manifests/train.csv"
    resolution: 256
    num_frames: 32

  # 优化
  optimizer: {type: adamw, lr: 2.0e-4, weight_decay: 0.01}
  scheduler: {type: cosine, warmup_steps: 5000}

  # 训练
  max_steps: 500000
  batch_size: 8                    # 每 GPU
  gradient_accumulation_steps: 2   # 有效 batch = 8 × 8 GPU × 2 = 128
  mixed_precision: bf16

  # 分布式
  distributed:
    backend: nccl
    fsdp_sharding_strategy: FULL_SHARD
    deepspeed_preset: "30b"        # 可选: "30b", "30b_moe", "200b", "4k_30s"
```

---

## 11. 附加设计特性

### 10.1 LipSync — 口型同步桥

**文件**: [seedance/models/mouth_roi_attention.py](seedance/models/mouth_roi_attention.py)

在 CBGA 之外添加额外的跨模态注意力，专注于口部区域：

- **MouthRegionMask**: 高斯注意力掩码，聚焦下脸部区域
- **MouthROIAttention**: 视频口部 token 查询音频 token 的交叉注意力
- **Viseme Embedding**: 14 类 MPEG-4 视位嵌入（bilabial, labiodental, dental 等）
- **音频→视位投影**: 从音频特征预测视位类别

### 10.2 人脸分析

**文件**: [seedance/models/face_analysis.py](seedance/models/face_analysis.py)

使用 InsightFace/MediaPipe 进行：
- 人脸检测 + 关键点
- 口部包围框（用于 LipSync 的精确掩码）

### 10.3 LFA Encoder — 身份特征锚点

**文件**: [seedance/models/lfa_encoder.py](seedance/models/lfa_encoder.py)

- 从参考帧提取全局角色身份特征
- 用于 SFT 训练中的角色一致性约束

### 10.4 KP Encoder — 3D 关键点编码器

**文件**: [seedance/models/kp_encoder.py](seedance/models/kp_encoder.py)

- 从面部关键点提取 3D 投影嵌入
- 作为分镜控制信号

### 10.5 Cascade Pipeline — 4K 30s 生成

通过三阶段级联实现超长超清视频生成，避免单次生成的计算爆炸。

---

## 12. 项目目录结构

```
xmedia-gen/
├── configs/                  # YAML 配置
│   ├── inference/            # T2VA/I2VA 推理配置
│   ├── model/                # 模型架构配置
│   └── train/                # 训练阶段配置
├── scripts/                  # 入口脚本
│   ├── train.py              # 训练启动（-m seedance.training）
│   ├── inference_t2va.py     # T2VA 推理
│   ├── inference_i2va.py     # I2VA 推理
│   ├── gradio_app.py         # Web 演示
│   └── build_balanced_manifest.py  # 均衡 manifest 构建
├── seedance/                 # 主包
│   ├── models/               # 模型定义
│   │   ├── video_vae/        # VideoVAE (encoder_3d, decoder_3d, causal_conv3d, discriminator_3d, lpips_loss, resnet3d)
│   │   ├── audio_vae/        # AudioVAE (encoder, decoder, mel_transform)
│   │   ├── db_dit/           # DB-DiT (db_dit, dual_branch_block, vision_branch, audio_branch, cross_modal_bridge, mm_rope, attention, adaln, moe, qk_norm, sparse_attention, cross_scale_attention, temporal_rope)
│   │   ├── text_encoder/     # T5 encoder
│   │   ├── common/           # Shared (embedding, layers, modulation, norm)
│   │   ├── face_analysis.py  # 人脸分析
│   │   ├── kp_encoder.py     # 3D 关键点编码器
│   │   ├── lfa_encoder.py    # LFA 身份编码器
│   │   ├── mouth_roi_attention.py  # 口型同步注意力
│   │   └── reward_model.py   # RLHF 奖励模型
│   ├── diffusion/            # 扩散框架
│   │   ├── flow_matching.py  # Flow Matching 训练+采样
│   │   ├── scheduler.py      # 推理调度器
│   │   ├── guidance.py       # CFG 引导
│   │   └── noise_schedule.py # LogitNormal 时间步采样
│   ├── data/                 # 数据管线
│   │   ├── video_dataset.py  # 视频数据集
│   │   ├── audio_dataset.py  # 音频数据集
│   │   ├── av_dataset.py     # AV 配对数据集
│   │   ├── bucket_sampler.py # 分桶采样器
│   │   ├── collate.py        # 批次整理
│   │   ├── transforms.py     # 数据增强
│   │   └── annotation/       # 自动标注 (scenario, motion, physics)
│   ├── training/             # 训练系统
│   │   ├── __main__.py       # 入口点
│   │   ├── trainer.py        # 基础训练器
│   │   ├── sft_trainer.py    # SFT 微调训练器
│   │   ├── rlhf_ppo.py       # RLHF PPO 训练器
│   │   ├── distributed.py    # FSDP/DDP/DeepSpeed 封装
│   │   ├── optimizer.py      # 优化器工厂
│   │   ├── lr_scheduler.py   # 学习率调度器
│   │   └── ema.py            # EMA 跟踪
│   ├── loss/                 # 损失函数
│   │   ├── flow_loss.py      # Flow Matching 损失
│   │   ├── vae_loss.py       # VAE 重建损失
│   │   ├── sync_loss.py      # AV 同步对比损失
│   │   ├── lip_sync_loss.py  # 口型同步损失
│   │   └── world_model_loss.py  # World Model 损失
│   ├── pipelines/            # 推理管线
│   │   ├── pipeline_t2va.py  # T2VA 管线
│   │   ├── pipeline_i2va.py  # I2VA 管线
│   │   └── pipeline_cascade.py  # 级联 4K 30s 管线
│   ├── physics/              # 物理探针
│   │   ├── physics_probe.py  # 物理学探针+损失
│   │   ├── causal_motion.py  # 因果运动分析
│   │   ├── phase_lock.py     # 相位锁定
│   │   ├── physcorr.py       # 物理相关性
│   │   └── vpt.py            # 视觉物理测试
│   ├── utils/                # 工具函数
│   │   ├── config.py         # 配置加载
│   │   ├── checkpoint.py     # 检查点保存/恢复
│   │   ├── logging.py        # 日志
│   │   ├── video_utils.py    # 视频 I/O
│   │   └── audio_utils.py    # 音频 I/O
│   └── tools/                # 数据准备 CLI 工具
│       ├── download_*.py     # 各类数据集下载
│       ├── build_manifest.py # Manifest 构建
│       ├── video_caption.py  # 自动标注
│       ├── scene_detection.py
│       ├── quality_filter.py
│       └── ...
├── tests/                    # 单元测试
│   ├── test_db_dit.py
│   ├── test_flow_matching.py
│   ├── test_video_vae.py
│   ├── test_mm_rope.py
│   ├── test_cbga.py
│   ├── test_annotation.py
│   ├── test_physics.py
│   └── test_pipeline.py
├── docs/                     # 文档
│   └── DESIGN.md             # 本设计文档
├── pyproject.toml            # 项目元数据和依赖
├── uv.lock                   # 锁定依赖
└── README.md                 # 项目说明
```

---

## 13. 测试策略

**文件**: [tests/](tests/)

| 测试文件 | 覆盖组件 |
|---------|---------|
| `test_db_dit.py` | DB-DiT 前后向传播、输出形状 |
| `test_flow_matching.py` | Flow Matching loss + 采样 + CFG |
| `test_video_vae.py` | VideoVAE 编解码、KL 损失、SDXL 初始化 |
| `test_mm_rope.py` | MM-RoPE 3D + 1D 旋转 |
| `test_cbga.py` | CBGA 门控、warmup、双向注意力 |
| `test_annotation.py` | 自动标注管线 |
| `test_physics.py` | Physics Probe 训练和推理 |
| `test_pipeline.py` | 端到端 T2VA / I2VA 管线 |

---

## 14. 性能参考

### 13.1 训练吞吐量

| 配置 | GPU | 每 GPU 批大小 | 有效批大小 | 步/秒 | 训练时间 (500K steps) |
|------|-----|-------------|-----------|-------|---------------------|
| Small | 1× A100 40GB | 4 | 4 | ~1.2 | ~5 天 |
| Base | 4× A100 80GB | 4 | 16 | ~0.8 | ~8 天 |
| Base | 8× A100 80GB | 8 | 128 | ~1.0 | ~6 天 |
| 30B MoE | 8× A100 80GB | 2 | 128 | ~0.3 | ~20 天 |

### 13.2 推理速度

| 配置 | 帧数 | 分辨率 | 采样步数 | 推理时间 (A100) |
|------|------|--------|---------|----------------|
| Base, Euler | 32 | 256×256 | 20 | ~15s |
| Base, Heun | 32 | 256×256 | 30 | ~40s |
| 30B, Heun | 32 | 256×256 | 30 | ~120s |
| Cascade 4K | 128 | 3840×2160 | 50 | ~10min |

---

## 15. 参考资料

- [Seedance 2.0: Advancing Video Generation for World Complexity](https://arxiv.org/abs/2604.14148) — ByteDance Seed Team, 2026
- [Seedance 1.5 Pro: A Native Audio-Visual Joint Generation Foundation Model](https://arxiv.org/abs/2512.13507) — ByteDance Seed Team, 2025
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., 2023
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Esser et al., 2024
- [Open-Sora: Democratizing Efficient Video Production](https://github.com/hpcaitech/Open-Sora) — HPC-AI Tech
- [DeepSeek-V2/MoE: A Strong, Economical Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — DeepSeek, 2024
- [The Invisible Hand of Physics in Video Diffusion Models](https://arxiv.org/abs/2606.xxxxx) — Esmati et al., 2026
