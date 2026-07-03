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
            model.enable_gradient_checkpointing = lambda: None
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
