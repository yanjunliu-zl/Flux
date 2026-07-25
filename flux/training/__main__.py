"""Entry point for Seedance training.

Supports single-GPU, single-node multi-GPU (DDP/FSDP), and multi-node (torchrun).

Usage:
    # Single GPU
    python -m flux.training --config configs/train/stage1_video_pretrain.yaml

    # Single-node multi-GPU (auto-detect)
    python -m flux.training --config configs/train/stage1_video_pretrain.yaml

    # Multi-node via torchrun
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
        --master_addr=192.168.1.1 --master_port=29500 \
        -m flux.training --config configs/train/stage1_video_pretrain.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from flux.models.db_dit import DBDiT
from flux.models.text_encoder import T5Encoder
from flux.training.trainer import Trainer
from flux.training.distributed import (
    setup_distributed,
    wrap_model,
    wrap_dataloader,
    all_reduce_losses,
    is_main_process,
)


def _build_from_config(cfg: dict, device: torch.device):
    """Instantiate a class from a config dict with _target_ key."""
    import importlib

    target = cfg.pop("_target_", None)
    if target is None:
        raise ValueError("Config must have '_target_' key for instantiation")

    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**cfg)


def main():
    # ── Hardware optimizations (before any tensor ops) ─────────────────
    if torch.cuda.is_available():
        # TF32 for matmul (Ampere+ GPUs) → ~1.3x speedup
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        # Auto-tune cuDNN kernels for fixed-size inputs
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="Seedance Training")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # ── Distributed setup (must happen first) ──────────────────────────
    local_rank, world_size, device = setup_distributed()

    # ── Load config ────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    training_cfg = full_config.get("training", full_config)
    stage = training_cfg.get("stage", "?")
    desc = training_cfg.get("description", "Seedance Training")

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"  Seedance Stage {stage}: {desc}")
        print(f"  Config: {args.config}")
        print(f"  World size: {world_size}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM/GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  Device: CPU")
        print(f"{'='*60}\n")

    # ── Dataset & DataLoader ───────────────────────────────────────────
    data_cfg = training_cfg.get("data", {})
    if not data_cfg:
        raise ValueError("Training config missing 'data' section")

    if is_main_process():
        print(f"[Data] Manifest: {data_cfg.get('manifest_path', '?')}")

    dataset = _build_from_config(data_cfg, device)

    if is_main_process():
        print(f"[Data] Dataset size: {len(dataset):,} samples")

    batch_size = training_cfg.get("batch_size", 4)
    num_workers = training_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(world_size <= 1),  # DistributedSampler sets shuffle=False
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
    )

    # Replace sampler with DistributedSampler when running multi-GPU
    train_loader = wrap_dataloader(train_loader, world_size, local_rank)

    if is_main_process():
        eff_batch = batch_size * world_size * training_cfg.get("gradient_accumulation_steps", 1)
        print(f"[Data] Batch: {batch_size}/GPU × {world_size} GPUs × "
              f"{training_cfg.get('gradient_accumulation_steps', 1)} accum = {eff_batch} effective")
        print(f"[Data] Workers: {num_workers}")

    # ── Model ──────────────────────────────────────────────────────────
    model_cfg = training_cfg.get("model", {})
    dim = model_cfg.get("dim", 1024)
    num_layers = model_cfg.get("num_layers", 24)
    num_heads = model_cfg.get("num_heads", 16)

    # Determine text encoder output dim (T5-base=768, T5-large=1024)
    text_cfg = training_cfg.get("text_encoder", {})
    text_model_name = text_cfg.get("model_name", "t5-base")
    # Map common T5 variants to output dims
    _T5_DIMS = {"t5-small": 512, "t5-base": 768, "t5-large": 1024, "t5-3b": 1024, "t5-11b": 1024}
    context_dim = _T5_DIMS.get(text_model_name, text_cfg.get("context_dim", 768))

    if context_dim != dim:
        if is_main_process():
            print(f"[Model] Text encoder dim ({context_dim}) != model dim ({dim}), "
                  f"cross-attention K/V will use context_dim")

    # CBGA in latter 50% of layers (overridable via config)
    cbga_start = model_cfg.get("cbga_start", num_layers // 2)
    cbga_layers = model_cfg.get("cbga_layers", list(range(cbga_start, num_layers)))

    model = DBDiT(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        context_dim=model_cfg.get("context_dim", context_dim),
        ffn_ratio=model_cfg.get("ffn_ratio", 4.0),
        qk_norm=model_cfg.get("qk_norm", True),
        dropout=model_cfg.get("dropout", 0.0),
        cbga_layers=cbga_layers,
        video_patch_size=tuple(model_cfg.get("video_patch_size", (1, 2, 2))),
        video_latent_channels=model_cfg.get("video_latent_channels", 16),
        audio_patch_size=tuple(model_cfg.get("audio_patch_size", (1, 4))),
        audio_latent_channels=model_cfg.get("audio_latent_channels", 8),
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main_process():
        print(f"[Model] DBDiT parameters: {total_params:.1f}M")

    # Load pretrained weights (before FSDP wrapping for efficiency)
    init_cfg = training_cfg.get("model_init", {})
    pretrained_path = init_cfg.get("pretrained_path", None)
    if pretrained_path and os.path.exists(pretrained_path):
        if is_main_process():
            print(f"[Model] Loading pretrained weights from {pretrained_path}...")
        state = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    # Wrap model with FSDP (or DDP fallback) for distributed training
    distributed_cfg = training_cfg.get("distributed", {})
    use_fsdp = distributed_cfg.get("fsdp_sharding_strategy", "FULL_SHARD") != "NONE"
    mixed_precision = training_cfg.get("mixed_precision", "bf16")
    grad_ckpt = training_cfg.get("gradient_checkpointing", True)

    model = wrap_model(
        model,
        mixed_precision=mixed_precision,
        activation_checkpointing=grad_ckpt,
        use_fsdp=use_fsdp,
    )
    if is_main_process():
        wrap_type = "FSDP" if (use_fsdp and world_size > 1) else ("DDP" if world_size > 1 else "none")
        print(f"[Model] Distributed wrap: {wrap_type}")

    # ── Text encoder (T5) ──────────────────────────────────────────────
    text_encoder = None
    if text_cfg.get("enabled", True):
        try:
            text_encoder = T5Encoder(
                model_name=text_cfg.get("model_name", "t5-base"),
                max_length=text_cfg.get("max_length", 77),
                device=device,
            )
            if is_main_process():
                print(f"[TextEncoder] T5 loaded: {text_cfg.get('model_name', 't5-base')}")
        except Exception as e:
            if is_main_process():
                print(f"[TextEncoder] T5 not available, using zero embeddings: {e}")
            text_encoder = None

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=training_cfg,
        device=device,
        text_encoder=text_encoder,
        val_loader=None,
        world_size=world_size,
        local_rank=local_rank,
    )

    if is_main_process():
        print(f"\n[Train] Starting: {training_cfg.get('max_steps', '?')} steps")
        print(f"[Train] Checkpoint dir: {trainer.checkpoint_dir}")
        if trainer.tb_writer is not None:
            print(f"[Train] TensorBoard: {trainer.tb_log_dir}")
        print()

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
