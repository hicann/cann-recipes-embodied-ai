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
export LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION="${LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION:-1}"
export LEROBOT_XVLA_USE_NPU_BADD_BMM="${LEROBOT_XVLA_USE_NPU_BADD_BMM:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$WORKSPACE_ROOT/lerobot}"
CONFIG_DIR="$(cd "$SCRIPT_DIR/../configs" && pwd)"
DEFAULT_STORAGE_ROOT="$WORKSPACE_ROOT"
if [[ -d /data/docker/xvla_storage ]]; then
    DEFAULT_STORAGE_ROOT=/data/docker/xvla_storage
fi
XVLA_STORAGE_ROOT="${XVLA_STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$XVLA_STORAGE_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$XVLA_STORAGE_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/transformers}"

NPROC=8
MASTER_PORT=29500
MODEL_TYPE=""
CUSTOM_CONFIG=""
USE_RESUME=false
USE_MIXED_PRECISION=false
MIXED_PRECISION_TYPE="bf16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc) NPROC="$2"; shift 2 ;;
        --port) MASTER_PORT="$2"; shift 2 ;;
        --config) CUSTOM_CONFIG="$2"; shift 2 ;;
        --resume) USE_RESUME=true; shift ;;
        --mix|--mixed|--mixed_precision)
            USE_MIXED_PRECISION=true
            if [[ -n "${2:-}" && ! "$2" =~ ^- ]]; then MIXED_PRECISION_TYPE="$2"; shift 2; else shift; fi
            ;;
        -h|--help)
            echo "Usage: $0 [--config <path>] [<model_type>] [--nproc <num>] [--port <port>] [--resume] [--mix [fp16|bf16]]"
            echo "Examples:"
            echo "  $0 xvla_libero --nproc 1 --port 29510"
            echo "  $0 xvla_libero --nproc 8 --port 29510"
            exit 0
            ;;
        *)
            if [[ -z "$MODEL_TYPE" ]]; then MODEL_TYPE="$1"; else echo "Unknown option or too many positional args: $1"; exit 1; fi
            shift
            ;;
    esac
done

if [[ -n "$CUSTOM_CONFIG" ]]; then
    if [[ "$CUSTOM_CONFIG" = /* ]]; then CONFIG_PATH="$CUSTOM_CONFIG"; else CONFIG_PATH="$RECIPE_REPO_ROOT/$CUSTOM_CONFIG"; fi
elif [[ -n "$MODEL_TYPE" ]]; then
    CONFIG_PATH="$CONFIG_DIR/${MODEL_TYPE}.yaml"
else
    echo "Error: Either --config <path> or <model_type> must be provided."
    exit 1
fi

[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH"; exit 1; }
[[ -d "$LEROBOT_ROOT" ]] || { echo "LeRobot repo not found: $LEROBOT_ROOT"; echo "Run: ./manipulation/xvla/train/src/scripts/setup.sh"; exit 1; }
command -v accelerate >/dev/null 2>&1 || { echo "accelerate not found, please activate the proper environment."; exit 1; }
command -v lerobot-train >/dev/null 2>&1 || { echo "lerobot-train not found, please activate the proper environment."; exit 1; }

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$XVLA_STORAGE_ROOT/ckpt/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${MODEL_TYPE:-custom}_${TIMESTAMP}.log"

RAW_OUTPUT_DIR=$(awk '/^[[:space:]]*output_dir:[[:space:]]*[^#]/ {gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); gsub(/[[:space:]]*#.*/, ""); gsub(/[[:space:]]+$/, ""); print; exit}' "$CONFIG_PATH")
RAW_JOB_NAME=$(awk '/^[[:space:]]*job_name:[[:space:]]*[^#]/ {gsub(/^[[:space:]]*job_name:[[:space:]]*/, ""); gsub(/[[:space:]]*#.*/, ""); gsub(/[[:space:]]+$/, ""); print; exit}' "$CONFIG_PATH")
if [[ -z "$RAW_OUTPUT_DIR" || -z "$RAW_JOB_NAME" ]]; then
    echo "Failed to parse output_dir or job_name from config: $CONFIG_PATH"
    exit 1
fi
OUTPUT_DIR_FINAL="${RAW_OUTPUT_DIR}_${TIMESTAMP}"
JOB_NAME_FINAL="${RAW_JOB_NAME}_${TIMESTAMP}"

TRAIN_ARGS=(--config_path="$CONFIG_PATH")
if [[ "$USE_RESUME" == true ]]; then TRAIN_ARGS+=(--resume=true); fi

ACCELERATE_ARGS=(--main_process_port "$MASTER_PORT" --num_processes "$NPROC")
if (( NPROC > 1 )); then ACCELERATE_ARGS+=(--multi_gpu); fi
if [[ "$USE_MIXED_PRECISION" == true ]]; then ACCELERATE_ARGS+=(--mixed_precision "$MIXED_PRECISION_TYPE"); fi

cd "$LEROBOT_ROOT"
nohup accelerate launch "${ACCELERATE_ARGS[@]}" \
    "$(command -v lerobot-train)" \
    "${TRAIN_ARGS[@]}" \
    --output_dir="$OUTPUT_DIR_FINAL" \
    --job_name="$JOB_NAME_FINAL" \
    > "$LOG_FILE" 2>&1 &
PID=$!

sleep 2
if ! ps -p "$PID" >/dev/null 2>&1; then
    echo "XVLA training failed to stay alive after launch."
    echo "Check log file: $LOG_FILE"
    exit 1
fi

echo "============================================="
echo "XVLA training started"
echo "LeRobot root: $LEROBOT_ROOT"
echo "Config: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo "Storage root: $XVLA_STORAGE_ROOT"
echo "Output dir: $OUTPUT_DIR_FINAL"
echo "Job name: $JOB_NAME_FINAL"
echo "Num processes: $NPROC"
echo "Master port: $MASTER_PORT"
echo "Resume: $USE_RESUME"
echo "NPU fusion attention: ${LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION}"
echo "NPU baddbmm domain linear: ${LEROBOT_XVLA_USE_NPU_BADD_BMM}"
echo "PID: $PID"
echo "============================================="
