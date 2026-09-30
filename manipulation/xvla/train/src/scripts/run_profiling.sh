#!/bin/bash
#
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
set -euo pipefail

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"
export HOST_CACHE_CAPACITY="${HOST_CACHE_CAPACITY:-20}"
export TOKENIZERS_PARALLELISM=false
export LEROBOT_XVLA_USE_NPU_BADD_BMM="${LEROBOT_XVLA_USE_NPU_BADD_BMM:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$WORKSPACE_ROOT/lerobot}"
DEFAULT_STORAGE_ROOT="$WORKSPACE_ROOT"
if [[ -d /data/docker/xvla_storage ]]; then
    DEFAULT_STORAGE_ROOT=/data/docker/xvla_storage
fi
XVLA_STORAGE_ROOT="${XVLA_STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"
LOG_DIR="$XVLA_STORAGE_ROOT/ckpt/logs"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$XVLA_STORAGE_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$XVLA_STORAGE_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/transformers}"

DTYPE="float16"
DEVICE="npu:0"
TRAIN_STEP=false
NUM_WARMUP=5
NUM_ITERS=20
BATCH_SIZE=2
SEQ_LEN=128
HIDDEN_SIZE=512
NUM_HEADS=8
DEPTH=4
CHUNK_SIZE=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --train-step) TRAIN_STEP=true; shift ;;
        --num-warmup) NUM_WARMUP="$2"; shift 2 ;;
        --num-iters) NUM_ITERS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
        --num-heads) NUM_HEADS="$2"; shift 2 ;;
        --depth) DEPTH="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--device npu:0] [--dtype float16|bfloat16|float32] [--train-step] [--num-warmup N] [--num-iters N] [--batch-size N] [--seq-len N] [--hidden-size N] [--num-heads N] [--depth N] [--chunk-size N]"
            echo "Examples:"
            echo "  $0 --dtype float16 --num-iters 20"
            echo "  $0 --dtype float32 --train-step --num-iters 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! -d "$LEROBOT_ROOT" ]]; then
    echo "LeRobot repo not found: $LEROBOT_ROOT"
    echo "Run: ./manipulation/xvla/train/src/scripts/setup.sh"
    exit 1
fi

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/profiling_xvla_kernels_${TIMESTAMP}.log"

ARGS=(
    --device "$DEVICE"
    --dtype "$DTYPE"
    --num_warmup "$NUM_WARMUP"
    --num_iters "$NUM_ITERS"
    --batch_size "$BATCH_SIZE"
    --seq_len "$SEQ_LEN"
    --hidden_size "$HIDDEN_SIZE"
    --num_heads "$NUM_HEADS"
    --depth "$DEPTH"
    --chunk_size "$CHUNK_SIZE"
)
if [[ "$TRAIN_STEP" == true ]]; then
    ARGS+=(--train_step)
fi

echo "============================================="
echo "XVLA kernel profiling started"
echo "LeRobot root: $LEROBOT_ROOT"
echo "Log file: $LOG_FILE"
echo "Storage root: $XVLA_STORAGE_ROOT"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "Train step: $TRAIN_STEP"
echo "Warmup/iters: $NUM_WARMUP/$NUM_ITERS"
echo "============================================="

cd "$LEROBOT_ROOT"
python "$SCRIPT_DIR/benchmark_xvla_npu_kernels.py" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
