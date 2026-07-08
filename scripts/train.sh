#!/usr/bin/env bash
# =============================================================================
# Seedance Training Launcher — single-GPU, multi-GPU, and multi-node.
# =============================================================================
#
# QUICK START (auto-detect hardware):
#   bash scripts/train.sh small            # 0.4B, 1× 24GB+
#   bash scripts/train.sh base             # 1.6B, 1× 48GB+
#   bash scripts/train.sh 30b              # 30.6B, 4× 80GB+
#   bash scripts/train.sh 30b_moe          # 28B MoE, 2× 80GB+
#   bash scripts/train.sh 200b             # 200B MoE, 16× 80GB+ (2-4 nodes)
#
# MANUAL (explicit config):
#   bash scripts/train.sh configs/train/stage1_video_pretrain.yaml
#   bash scripts/train.sh configs/train/stage1_video_pretrain.yaml 8
#   bash scripts/train.sh configs/train/stage1_video_pretrain.yaml 8 4 0 192.168.1.1
#
# Args:
#   $1: config path, OR one of: small | base | 30b | 30b_moe | 200b
#   $2: GPUs per node (default: auto-detect)
#   $3: total nodes (default: 1)
#   $4: node rank (default: 0)
#   $5: master address (default: localhost)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# ── Preset → config + hardware requirements ──────────────────────────
declare -A PRESET_CONFIG PRESET_MIN_GPU PRESET_MIN_GPUS PRESET_MIN_RAM PRESET_NOTE

PRESET_CONFIG[small]="configs/train/stage1_video_pretrain.yaml"
PRESET_MIN_GPU[small]="24"
PRESET_MIN_GPUS[small]="1"
PRESET_MIN_RAM[small]="64"
PRESET_NOTE[small]="single RTX 3090/4090, auto-detect works"

PRESET_CONFIG[base]="configs/train/stage1_video_pretrain.yaml"
PRESET_MIN_GPU[base]="48"
PRESET_MIN_GPUS[base]="1"
PRESET_MIN_RAM[base]="128"
PRESET_NOTE[base]="single A6000/L40S/RTX PRO 6000"

PRESET_CONFIG[30b]="configs/train/stage1_30b.yaml"
PRESET_MIN_GPU[30b]="80"
PRESET_MIN_GPUS[30b]="4"
PRESET_MIN_RAM[30b]="512"
PRESET_NOTE[30b]="4× A100 (80GB), FSDP FULL_SHARD"

PRESET_CONFIG[30b_moe]="configs/train/stage1_30b_moe.yaml"
PRESET_MIN_GPU[30b_moe]="80"
PRESET_MIN_GPUS[30b_moe]="2"
PRESET_MIN_RAM[30b_moe]="256"
PRESET_NOTE[30b_moe]="2× A100 (80GB), FSDP FULL_SHARD + CPU offload"

PRESET_CONFIG[200b]="configs/train/stage1_200b_moe.yaml"
PRESET_MIN_GPU[200b]="80"
PRESET_MIN_GPUS[200b]="16"
PRESET_MIN_RAM[200b]="2048"
PRESET_NOTE[200b]="16× A100/H100 (80GB) across 2-4 nodes; optimizer states require 1.6TB CPU RAM total"

# ── Parse arguments ───────────────────────────────────────────────────
ARG1="${1:-}"
if [[ -n "$ARG1" && -n "${PRESET_CONFIG[$ARG1]:-}" ]]; then
    CONFIG="${PRESET_CONFIG[$ARG1]}"
    PRESET="$ARG1"
    GPUS_PER_NODE="${2:-}"
    NNODES="${3:-0}"      # 0 = auto-calculate from min GPUs
    NODE_RANK="${4:-0}"
    MASTER_ADDR="${5:-localhost}"

    # Auto-detect GPUs if not specified
    if [ -z "$GPUS_PER_NODE" ]; then
        GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
        [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
    fi

    # Calculate required nodes: ceil(min_gpus / gpus_per_node)
    if [ "$NNODES" -eq 0 ]; then
        NEED=${PRESET_MIN_GPUS[$ARG1]}
        NNODES=$(( (NEED + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    fi
else
    CONFIG="$ARG1"
    PRESET=""
    GPUS_PER_NODE="${2:-}"
    NNODES="${3:-1}"
    NODE_RANK="${4:-0}"
    MASTER_ADDR="${5:-localhost}"
fi

if [ -z "$CONFIG" ]; then
    echo -e "${RED}Usage: $0 <config.yaml|preset> [gpus_per_node] [nnodes] [node_rank] [master_addr]${NC}"
    echo ""
    echo "Presets:"
    echo "  small       0.4B model, 1× 24GB GPU"
    echo "  base        1.6B model, 1× 48GB GPU"
    echo "  30b         30.6B dense, 4× 80GB GPU"
    echo "  30b_moe     28B MoE (10B activated), 2× 80GB GPU"
    echo "  200b        200B MoE (36B activated), 16× 80GB GPU (2-4 nodes)"
    exit 1
fi

MASTER_PORT="${MASTER_PORT:-29500}"

# ── Auto-detect GPU count and type (for non-preset mode) ─────────────
if [ -z "${PRESET:-}" ] && [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
    [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
fi

GPU_NAME=""
GPU_MEM=0
if command -v nvidia-smi &>/dev/null && nvidia-smi --list-gpus &>/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_MEM=$(( GPU_MEM / 1024 ))
fi

TOTAL_GPUS=$(( GPUS_PER_NODE * NNODES ))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Print banner ──────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  ${GREEN}Seedance Training Launcher${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}  Config:       $CONFIG"
echo -e "${CYAN}║${NC}  GPUs/node:    $GPUS_PER_NODE"
echo -e "${CYAN}║${NC}  Nodes:        $NNODES (rank $NODE_RANK)"
echo -e "${CYAN}║${NC}  Total GPUs:   $TOTAL_GPUS"
echo -e "${CYAN}║${NC}  Master:       $MASTER_ADDR:$MASTER_PORT"
echo -e "${CYAN}║${NC}  Hardware:     ${GPU_MEM}GB ${GPU_NAME:-unknown GPU}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Hardware validation ───────────────────────────────────────────────
if [ -n "$PRESET" ]; then
    NEED_GPU="${PRESET_MIN_GPU[$PRESET]}"
    NEED_GPUS="${PRESET_MIN_GPUS[$PRESET]}"
    NEED_RAM="${PRESET_MIN_RAM[$PRESET]}"

    WARNINGS=0

    if [ "$GPU_MEM" -gt 0 ] && [ "$GPU_MEM" -lt "$NEED_GPU" ]; then
        echo -e "  ${RED}⚠ VRAM:${NC}    each GPU has ${GPU_MEM}GB, preset needs ${NEED_GPU}GB+"
        WARNINGS=$((WARNINGS+1))
    fi

    if [ "$TOTAL_GPUS" -lt "$NEED_GPUS" ]; then
        echo -e "  ${RED}⚠ GPUs:${NC}    have $TOTAL_GPUS total, preset needs $NEED_GPUS+ (add nodes or GPUs)"
        WARNINGS=$((WARNINGS+1))
    fi

    TOTAL_RAM=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    if [ "$TOTAL_RAM" -gt 0 ] && [ "$TOTAL_RAM" -lt "$NEED_RAM" ]; then
        echo -e "  ${YELLOW}⚠ RAM:${NC}    system has ${TOTAL_RAM}GB, preset recommends ${NEED_RAM}GB+"
        WARNINGS=$((WARNINGS+1))
    fi

    if [ "$WARNINGS" -gt 0 ]; then
        echo ""
        echo -e "  ${RED}Hardware below minimum for this preset.${NC}"
        echo -e "  ${YELLOW}Expected:${NC} ${PRESET_NOTE[$PRESET]}"
        echo -e "  ${YELLOW}Detected:${NC} ${GPU_MEM}GB × $TOTAL_GPUS = $(( GPU_MEM * TOTAL_GPUS ))GB total VRAM"
        echo ""
        # Skip prompt in non-interactive mode (CI, pipelines)
        if [ -t 0 ]; then
            read -rp "  Continue anyway? [y/N] " yn
            case $yn in
                [Yy]*) echo "" ;;
                *)     exit 1 ;;
            esac
        else
            echo -e "  ${RED}Non-interactive mode — aborting (redirect or set GPUS/NODES explicitly).${NC}"
            exit 1
        fi
    else
        echo -e "  ${GREEN}✓ Hardware check passed${NC} (${PRESET_NOTE[$PRESET]})"
        echo ""
    fi
fi

# ── Launch ────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

if [ "$NNODES" -gt 1 ] || [ "$GPUS_PER_NODE" -gt 1 ]; then
    # Multi-GPU / Multi-node via torchrun
    echo -e "  Launching: torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES ..."
    echo ""

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
    echo -e "  Launching: python -m seedance.training --config $CONFIG"
    echo ""
    python -m seedance.training --config "$CONFIG"
fi
