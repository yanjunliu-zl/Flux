#!/usr/bin/env bash
# Seedance training launcher — supports single-GPU, multi-GPU, and multi-node.
#
# Usage:
#   # Single GPU (auto-detect)
#   bash scripts/train.sh configs/train/stage1_video_pretrain.yaml
#
#   # Multi-GPU single node (auto-detect all GPUs)
#   bash scripts/train.sh configs/train/stage1_video_pretrain.yaml 8
#
#   # Multi-node (via torchrun)
#   bash scripts/train.sh configs/train/stage1_video_pretrain.yaml 8 2 0 192.168.1.1
#
# Args:
#   $1: config path (required)
#   $2: GPUs per node (default: all available, or 1 if nvidia-smi fails)
#   $3: total nodes (default: 1)
#   $4: node rank (0-indexed, default: 0)
#   $5: master address (default: localhost)

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [gpus_per_node] [nnodes] [node_rank] [master_addr]}"
GPUS_PER_NODE="${2:-}"
NNODES="${3:-1}"
NODE_RANK="${4:-0}"
MASTER_ADDR="${5:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Auto-detect GPU count if not specified
if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
    if [ "$GPUS_PER_NODE" -eq 0 ]; then
        GPUS_PER_NODE=1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  Seedance Training Launcher"
echo "  Config:       $CONFIG"
echo "  GPUs/node:    $GPUS_PER_NODE"
echo "  Nodes:        $NNODES"
echo "  Node rank:    $NODE_RANK"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT"
echo "============================================"

if [ "$NNODES" -gt 1 ] || [ "$GPUS_PER_NODE" -gt 1 ]; then
    # Multi-GPU / Multi-node: use torchrun
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        -m seedance.training \
        --config "$CONFIG"
else
    # Single GPU
    python -m seedance.training --config "$CONFIG"
fi
