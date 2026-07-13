"""Distributed training setup (FSDP, DDP, mixed precision)."""

import os
import torch
import torch.nn as nn
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed training environment.

    Supports both single-node and multi-node via torchrun:
        torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
            --master_addr=192.168.1.1 --master_port=29500 \
            -m seedance.training --config configs/train/stage1.yaml

    Returns:
        Tuple of (local_rank, world_size, device).
    """
    # Multi-node / torchrun: RANK, WORLD_SIZE, LOCAL_RANK set by launcher
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
    # Single-node multi-GPU: auto-detect all visible GPUs
    elif torch.cuda.device_count() > 1:
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"[Distributed] world_size={world_size}, local_rank={local_rank}")
    return local_rank, world_size, device


def wrap_model(
    model: nn.Module,
    mixed_precision: str = "bf16",
    activation_checkpointing: bool = True,
    use_fsdp: bool = True,
) -> nn.Module:
    """Wrap model with FSDP or DDP for distributed training.

    Args:
        model: The model to wrap.
        mixed_precision: "bf16", "fp16", or "fp32".
        activation_checkpointing: Whether to use gradient checkpointing.
        use_fsdp: If True, use FSDP; otherwise use DDP.

    Returns:
        Wrapped model.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size <= 1:
        if activation_checkpointing:
            # Apply gradient checkpointing manually for single GPU
            # Without this, 32fr×256px×batch8 activations can hit 80GB+
            from torch.utils.checkpoint import checkpoint
            from seedance.models.db_dit.dual_branch_block import DualBranchBlock
            for module in model.modules():
                if isinstance(module, DualBranchBlock):
                    module._grad_ckpt = True
        return model

    if use_fsdp:
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                BackwardPrefetch,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.wrap import (
                transformer_auto_wrap_policy,
            )
            from functools import partial
            from seedance.models.db_dit.dual_branch_block import DualBranchBlock

            mp_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }.get(mixed_precision, torch.bfloat16)

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={DualBranchBlock},
            )

            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=MixedPrecision(
                    param_dtype=mp_dtype,
                    reduce_dtype=mp_dtype,
                    buffer_dtype=mp_dtype,
                ),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=torch.cuda.current_device(),
                use_orig_params=True,
            )

            if activation_checkpointing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                    CheckpointImpl,
                )
                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=checkpoint_wrapper,
                    auto_wrap_policy=auto_wrap_policy,
                )

        except ImportError:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
            )
    else:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
        )

    return model


def is_main_process() -> bool:
    """Check if current process is the main one."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def wrap_dataloader(loader, world_size: int, rank: int):
    """Replace the loader's sampler with a DistributedSampler.

    Call ``loader.sampler.set_epoch(epoch)`` each epoch to shuffle differently
    across ranks.

    Args:
        loader: DataLoader to wrap (modified in-place).
        world_size: Total number of processes.
        rank: Rank of current process.

    Returns:
        The modified DataLoader.
    """
    from torch.utils.data.distributed import DistributedSampler

    if world_size <= 1:
        return loader

    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    # Replace the loader's sampler; keep other settings
    loader.sampler = sampler
    # shuffle must be False when sampler is used
    loader.shuffle = False
    loader.drop_last = True
    return loader


def all_reduce_losses(losses: dict[str, float]) -> dict[str, float]:
    """Average loss values across all distributed processes.

    Args:
        losses: Dict of loss names to scalar values (on current device).

    Returns:
        The same dict with values averaged across all ranks.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return losses

    world_size = dist.get_world_size()
    for k in losses:
        t = torch.tensor(losses[k], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        losses[k] = t.item()
    return losses


# ═══════════════════════════════════════════════════════════════════════
# DeepSpeed Backend (optional — used for 200B+ or 4K 30s training)
# ═══════════════════════════════════════════════════════════════════════

def _build_deepspeed_config(
    model_size: str = "30b",
    zero_stage: int = 3,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    offload_nvme: bool = False,
    nvme_path: str = "/tmp/deepspeed_offload",
    sequence_parallel: bool = False,
) -> dict:
    """Build a DeepSpeed ZeRO configuration dict.

    Args:
        model_size: "base", "30b", "200b" — tunes batch/offload settings.
        zero_stage: ZeRO stage (1, 2, or 3). Stage 3 = FSDP equivalent.
        offload_optimizer: Offload optimizer states to CPU RAM.
        offload_param: Offload model params to CPU RAM.
        offload_nvme: Offload to NVMe SSD (for 200B where CPU RAM isn't enough).
        nvme_path: NVMe offload directory.
        sequence_parallel: Enable Ulysses sequence parallelism (4K 30s).

    Returns:
        DeepSpeed config dict (can be saved as JSON or passed directly).
    """
    # ── Per-size defaults ──────────────────────────────────────────
    size_defaults = {
        "base": {"train_batch_size": 128, "grad_accum": 1},
        "30b":  {"train_batch_size": 128, "grad_accum": 4},
        "200b": {"train_batch_size": 512, "grad_accum": 64},
    }
    sd = size_defaults.get(model_size, size_defaults["30b"])

    cfg = {
        "train_batch_size": sd["train_batch_size"],
        "gradient_accumulation_steps": sd["grad_accum"],
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none",
                # NVMe offload for optimizer states (200B)
                "nvme_path": nvme_path if offload_nvme else None,
            } if offload_optimizer else None,
            "offload_param": {
                "device": "cpu" if offload_param else "none",
                "nvme_path": nvme_path if offload_nvme else None,
            } if offload_param else None,
            # Stage 3 specific
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
        },
        "gradient_clipping": 1.0,
    }

    # Clean up None values
    if cfg["zero_optimization"]["offload_optimizer"] is None:
        del cfg["zero_optimization"]["offload_optimizer"]
    if cfg["zero_optimization"].get("offload_param") is None:
        del cfg["zero_optimization"]["offload_param"]

    # Remove NVMe path if not using NVMe offload
    if not offload_nvme:
        for key in ("offload_optimizer", "offload_param"):
            if key in cfg["zero_optimization"]:
                offload = cfg["zero_optimization"][key]
                if isinstance(offload, dict) and "nvme_path" in offload:
                    del offload["nvme_path"]

    if sequence_parallel:
        cfg["sequence_parallel"] = {"enabled": True}

    return cfg


# ── Pre-built configs for each model size ─────────────────────────────

def get_deepspeed_config_30b() -> dict:
    """30B dense: ZeRO-3, no offload needed (4× A100 80GB)."""
    return _build_deepspeed_config("30b", zero_stage=3, offload_optimizer=False)

def get_deepspeed_config_30b_moe() -> dict:
    """30B MoE: ZeRO-3, CPU optimizer offload (expert params are large)."""
    return _build_deepspeed_config("30b", zero_stage=3, offload_optimizer=True)

def get_deepspeed_config_200b() -> dict:
    """200B MoE: ZeRO-3 + NVMe offload for optimizer states."""
    return _build_deepspeed_config(
        "200b", zero_stage=3,
        offload_optimizer=True, offload_nvme=True,
    )

def get_deepspeed_config_4k_30s() -> dict:
    """4K 30s: ZeRO-3 + sequence parallelism for long sequences."""
    return _build_deepspeed_config(
        "200b", zero_stage=3,
        offload_optimizer=True, offload_nvme=True,
        sequence_parallel=True,
    )


def wrap_model_deepspeed(
    model: nn.Module,
    config: dict | str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> "DeepSpeedEngine":
    """Wrap model with DeepSpeed ZeRO engine.

    Args:
        model: The DB-DiT model (on CPU, before device placement).
        config: DeepSpeed config dict OR preset name
                ("30b", "30b_moe", "200b", "4k_30s").
        optimizer: Optional PyTorch optimizer (DeepSpeed can create its own).
        scheduler: Optional LR scheduler.

    Returns:
        DeepSpeedEngine (model + optimizer + scheduler all wrapped).

    Raises:
        ImportError if deepspeed is not installed.
    """
    try:
        import deepspeed
    except ImportError:
        raise ImportError(
            "DeepSpeed is not installed. Install with:\n"
            "  uv pip install deepspeed\n"
            "  or: pip install deepspeed"
        )

    # ── Resolve config preset ───────────────────────────────────────
    if isinstance(config, str):
        preset_map = {
            "30b": get_deepspeed_config_30b,
            "30b_moe": get_deepspeed_config_30b_moe,
            "200b": get_deepspeed_config_200b,
            "4k_30s": get_deepspeed_config_4k_30s,
        }
        if config in preset_map:
            config = preset_map[config]()
        else:
            raise ValueError(
                f"Unknown DeepSpeed preset: {config}. "
                f"Choose from: {list(preset_map.keys())}"
            )

    # ── Create engine ───────────────────────────────────────────────
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config_params=config,
        model_parameters=model.parameters(),
    )

    return engine


def estimate_deepspeed_memory(model_size: str) -> None:
    """Estimate per-GPU memory with DeepSpeed at different ZeRO stages.

    Args:
        model_size: "base", "30b", "200b".
    """
    params_map = {"base": 1.6e9, "30b": 30.6e9, "200b": 200e9}
    params = params_map.get(model_size, 30.6e9)

    print(f"\n  DeepSpeed memory estimate for {model_size} ({params/1e9:.0f}B params):\n")
    print(f"  {'Stage':<12} {'Model/GPU':>10} {'Grad/GPU':>10} {'Opt/GPU':>10} {'Total/GPU':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    gpu_counts = [1, 2, 4, 8, 16]
    for stage, offload in [(2, "none"), (3, "none"), (3, "cpu"), (3, "nvme")]:
        for n in gpu_counts:
            model_gb = params * 2 / n / 1e9
            grad_gb = params * 2 / n / 1e9
            if offload == "none":
                opt_gb = params * 12 / n / 1e9  # Adam m+v+fp32 master
            elif offload == "cpu":
                opt_gb = 0  # offloaded to CPU
            else:
                opt_gb = 0  # offloaded to NVMe

            total = model_gb + grad_gb + opt_gb + 15  # +15GB activations
            label = f"Z{stage}+{offload}"
            print(f"  {label:<12} {model_gb:>8.0f}GB {grad_gb:>8.0f}GB {opt_gb:>8.0f}GB {total:>8.0f}GB  {'✓' if total < 80 else '✗'}")
        print()

