#!/bin/bash
# Copyright (c) 2026 Institute of Software, Chinese Academy of Sciences (ISCAS). All rights reserved.
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================
# run_train.sh
# 功能: 通用训练脚本，根据模型名自动加载配置文件
# 支持后台运行、resume模式、混合精度模式、日志管理
# =============================================

# export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:32"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000
export HOST_CACHE_CAPACITY=20
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PI05_FUSE_PALIGEMMA_QKV="${PI05_FUSE_PALIGEMMA_QKV:-1}"
export PI05_USE_NPU_FUSION_ATTENTION="${PI05_USE_NPU_FUSION_ATTENTION:-1}"
export PI05_USE_NPU_GROUPED_GEMMA_INPROJ="${PI05_USE_NPU_GROUPED_GEMMA_INPROJ:-1}"
export LEROBOT_DDP_FIND_UNUSED_PARAMETERS="${LEROBOT_DDP_FIND_UNUSED_PARAMETERS:-auto}"
export LEROBOT_DDP_STATIC_GRAPH="${LEROBOT_DDP_STATIC_GRAPH:-auto}"
export LEROBOT_DDP_GRADIENT_AS_BUCKET_VIEW="${LEROBOT_DDP_GRADIENT_AS_BUCKET_VIEW:-auto}"

# -------------------------
# 获取项目根目录
# -------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEFAULT_PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../../../../" && pwd )/lerobot"
PROJECT_ROOT="$DEFAULT_PROJECT_ROOT"
CONDA_ENV_NAME="${LEROBOT_ENV_NAME:-lerobot}"
ACCELERATE_BIN=""
LEROBOT_TRAIN_BIN=""

resolve_lerobot_bins() {
    local candidate_dirs=()
    local python_bin_dir=""
    local conda_base=""
    local dir=""

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        candidate_dirs+=("${CONDA_PREFIX}/bin")
    fi

    if command -v python >/dev/null 2>&1; then
        python_bin_dir=$(python - <<'EOF'
import os
import sys
print(os.path.dirname(sys.executable))
EOF
)
        if [[ -n "${python_bin_dir}" ]]; then
            candidate_dirs+=("${python_bin_dir}")
        fi
    fi

    candidate_dirs+=("${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin")
    candidate_dirs+=("${HOME}/anaconda3/envs/${CONDA_ENV_NAME}/bin")

    if command -v conda >/dev/null 2>&1; then
        conda_base=$(conda info --base 2>/dev/null || true)
        if [[ -n "${conda_base}" ]]; then
            candidate_dirs+=("${conda_base}/envs/${CONDA_ENV_NAME}/bin")
        fi
    fi

    for dir in "${candidate_dirs[@]}"; do
        if [[ -x "${dir}/accelerate" && -x "${dir}/lerobot-train" ]]; then
            ACCELERATE_BIN="${dir}/accelerate"
            LEROBOT_TRAIN_BIN="${dir}/lerobot-train"
            return 0
        fi
    done

    if command -v accelerate >/dev/null 2>&1 && command -v lerobot-train >/dev/null 2>&1; then
        ACCELERATE_BIN=$(command -v accelerate)
        LEROBOT_TRAIN_BIN=$(command -v lerobot-train)
        return 0
    fi

    return 1
}

# -------------------------
# 默认参数
# -------------------------
NPROC=8
MASTER_PORT=29500
MODEL_TYPE=""
CUSTOM_CONFIG=""
CUSTOM_LEROBOT_DIR=""
USE_RESUME=false
USE_MIXED_PRECISION=false
MIXED_PRECISION_TYPE="bf16"  # 默认使用 bf16
RUN_FOREGROUND=false
DISABLE_OUTER_TRAIN_SUFFIX_CHECKPOINT=false
FORCE_GRADIENT_CHECKPOINTING=""

# -------------------------
# 参数解析
# -------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        --lerobot-dir)
            CUSTOM_LEROBOT_DIR="$2"
            shift 2
            ;;
        --resume)
            USE_RESUME=true
            shift
            ;;
        --foreground)
            RUN_FOREGROUND=true
            shift
            ;;
        --mix|--mixed|--mixed_precision)
            USE_MIXED_PRECISION=true
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                MIXED_PRECISION_TYPE="$2"
                shift 2
            else
                MIXED_PRECISION_TYPE="fp16"
                shift 1
            fi
            ;;
        --gc)
            FORCE_GRADIENT_CHECKPOINTING=true
            shift
            ;;
        --disable-outer-suffix-checkpoint)
            DISABLE_OUTER_TRAIN_SUFFIX_CHECKPOINT=true
            shift
            ;;
        --disable-gc)
            FORCE_GRADIENT_CHECKPOINTING=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--config <path>] [--lerobot-dir <path>] [<model_type>] [--nproc <num>] [--port <port>] [--resume] [--foreground] [--gc|--disable-gc] [--disable-outer-suffix-checkpoint] [--mix [fp16|bf16|fp8]]"
            echo "Example:"
            echo "  ./run_train.sh pi05"
            echo "  ./run_train.sh pi05 --resume"
            echo "  ./run_train.sh pi05 --mix fp16"
            echo "  ./run_train.sh --config configs/custom.yaml --mix bf16"
            echo "  ./run_train.sh pi05 --lerobot-dir /path/to/lerobot-v044"
            echo "  ./run_train.sh pi05 --nproc 1 --foreground"
            echo "  ./run_train.sh pi05 --nproc 2 --disable-gc"
            echo "  ./run_train.sh pi05 --nproc 2 --disable-outer-suffix-checkpoint"
            exit 0
            ;;
        *)
            if [ -z "$MODEL_TYPE" ]; then
                MODEL_TYPE="$1"
            else
                echo "Unknown option or too many positional args: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -n "$CUSTOM_LEROBOT_DIR" ]]; then
    PROJECT_ROOT="$CUSTOM_LEROBOT_DIR"
fi
echo "Project root: $PROJECT_ROOT"

# -------------------------
# 确定配置文件路径
# -------------------------
if [ -n "$CUSTOM_CONFIG" ]; then
    if [[ "$CUSTOM_CONFIG" = /* ]]; then
        CONFIG_PATH="$CUSTOM_CONFIG"
    else
        CONFIG_PATH="$PROJECT_ROOT/$CUSTOM_CONFIG"
    fi
    echo "Using custom config: $CONFIG_PATH"
elif [ -n "$MODEL_TYPE" ]; then
    if [[ "$MODEL_TYPE" != "pi05" ]]; then
        echo "Error: Unsupported model type '$MODEL_TYPE'. Only 'pi05' is supported; use --config <path> for a custom config."
        exit 1
    fi
    CONFIG_PATH="$PROJECT_ROOT/src/lerobot/configs/${MODEL_TYPE}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Config file not found for model '$MODEL_TYPE': $CONFIG_PATH"
        exit 1
    fi
    echo "Using config for model '$MODEL_TYPE': $CONFIG_PATH"
else
    echo "Error: Either --config <path> or <model_type> must be provided."
    exit 1
fi

if [[ -n "$FORCE_GRADIENT_CHECKPOINTING" ]]; then
    OVERRIDDEN_CONFIG_PATH=$(mktemp "/tmp/${MODEL_TYPE:-custom}_gc_override_XXXXXX.yaml")
    trap 'rm -f "$OVERRIDDEN_CONFIG_PATH"' EXIT
    cp "$CONFIG_PATH" "$OVERRIDDEN_CONFIG_PATH"
    python - "$OVERRIDDEN_CONFIG_PATH" "$FORCE_GRADIENT_CHECKPOINTING" <<'EOF'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
forced_value = sys.argv[2].lower()
text = config_path.read_text()
pattern = re.compile(r'^(\s*gradient_checkpointing:\s*)(true|false)(\s*)$', re.MULTILINE)
match = pattern.search(text)
if match is None:
    raise SystemExit(f'gradient_checkpointing not found in {config_path}')
text = pattern.sub(lambda m: f"{m.group(1)}{forced_value}{m.group(3)}", text, count=1)
config_path.write_text(text)
EOF
    CONFIG_PATH="$OVERRIDDEN_CONFIG_PATH"
    echo "Overriding policy.gradient_checkpointing: $FORCE_GRADIENT_CHECKPOINTING"
    echo "Using temporary config: $CONFIG_PATH"
fi

# -------------------------
# 定位训练环境中的 accelerate 与 lerobot-train
# -------------------------
if ! resolve_lerobot_bins; then
    echo "Error: 未找到可用的 accelerate 与 lerobot-train。请先激活 lerobot 环境，或通过环境变量 LEROBOT_ENV_NAME 指定 conda 环境名。"
    exit 1
fi

echo "Using accelerate: $ACCELERATE_BIN"
echo "Using lerobot-train: $LEROBOT_TRAIN_BIN"

if [[ "$DISABLE_OUTER_TRAIN_SUFFIX_CHECKPOINT" == true ]]; then
    export PI05_DISABLE_OUTER_TRAIN_SUFFIX_CHECKPOINT=1
else
    export PI05_DISABLE_OUTER_TRAIN_SUFFIX_CHECKPOINT="${PI05_DISABLE_OUTER_TRAIN_SUFFIX_CHECKPOINT:-0}"
fi


# -------------------------
# 检查 output_dir 与 resume
# -------------------------
OUTPUT_DIR_ORIG=$(awk '/^[[:space:]]*output_dir:/{gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")

if [[ -n "$OUTPUT_DIR_ORIG" ]]; then
    if [[ "$OUTPUT_DIR_ORIG" != /* ]]; then
        OUTPUT_DIR_ORIG="$PROJECT_ROOT/$OUTPUT_DIR_ORIG"
    fi
fi

RESUME_IN_CONFIG=$(awk '/^[[:space:]]*resume:/{gsub(/^[[:space:]]*resume:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH" | tr '[:upper:]' '[:lower:]')

if [[ "$USE_RESUME" == true ]]; then
    RESUME_FINAL=true
elif [[ "$RESUME_IN_CONFIG" == "true" ]]; then
    RESUME_FINAL=true
else
    RESUME_FINAL=false
fi

if [[ "$RESUME_FINAL" != true ]] && [[ -n "$OUTPUT_DIR_ORIG" ]] && [[ -d "$OUTPUT_DIR_ORIG" ]]; then
    OUTPUT_DIR_NEW="${OUTPUT_DIR_ORIG}_$(date +%Y%m%d_%H%M%S)"
    echo "Auto-adjusted output_dir to avoid conflict: $OUTPUT_DIR_NEW"
fi

# -------------------------
# 日志设置
# -------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/ckpt/logs/train_${MODEL_TYPE:-custom}_${TIMESTAMP}.log"
mkdir -p "$PROJECT_ROOT/ckpt/logs"

# -------------------------
# 构建训练命令参数
# -------------------------
TRAIN_ARGS=(--config_path="$CONFIG_PATH")

if [[ "$USE_RESUME" == true ]]; then
    TRAIN_ARGS+=(--resume=true)
    echo "Resume mode enabled."
fi

# -------------------------
# 构建 accelerate 参数
# -------------------------
ACCELERATE_MIXED_PRECISION="no"
ACCELERATE_ARGS=(--num_machines 1 --main_process_port "$MASTER_PORT" --dynamo_backend no)
if (( NPROC != 1 )); then
    ACCELERATE_ARGS+=(--multi_gpu --num_processes "$NPROC")
fi

if [[ "$USE_MIXED_PRECISION" == true ]]; then
    echo "Using mixed precision: $MIXED_PRECISION_TYPE"
    ACCELERATE_MIXED_PRECISION="$MIXED_PRECISION_TYPE"
fi
ACCELERATE_ARGS+=(--mixed_precision "$ACCELERATE_MIXED_PRECISION")

# -------------------------
# 启动训练
# -------------------------
RAW_OUTPUT_DIR=$(awk '/^[[:space:]]*output_dir:/{gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
RAW_JOB_NAME=$(awk '/^[[:space:]]*job_name:/{gsub(/^[[:space:]]*job_name:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
OUTPUT_DIR_FINAL="${RAW_OUTPUT_DIR}_${TIMESTAMP}"
JOB_NAME_FINAL="${RAW_JOB_NAME}_${TIMESTAMP}"

TRAIN_CMD=(
    "$ACCELERATE_BIN" launch "${ACCELERATE_ARGS[@]}"
    "$LEROBOT_TRAIN_BIN"
    "${TRAIN_ARGS[@]}"
    --output_dir="$OUTPUT_DIR_FINAL"
    --job_name="$JOB_NAME_FINAL"
)

echo ""
echo "============================================="
echo "Training started for model: ${MODEL_TYPE:-custom}"
echo "Config file: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo "Foreground: $RUN_FOREGROUND"
echo "Mixed precision: $ACCELERATE_MIXED_PRECISION"
echo "Resume: $RESUME_FINAL"
echo "Num processes: $NPROC"
echo "PI05 disable outer train suffix checkpoint: $PI05_DISABLE_OUTER_TRAIN_SUFFIX_CHECKPOINT"
echo "PI05 fused PaliGemma qkv: $PI05_FUSE_PALIGEMMA_QKV"
echo "PI05 fused attention: $PI05_USE_NPU_FUSION_ATTENTION"
echo "Gradient checkpointing override: ${FORCE_GRADIENT_CHECKPOINTING:-config_default}"
echo "DDP find_unused_parameters: $LEROBOT_DDP_FIND_UNUSED_PARAMETERS"
echo "DDP static_graph: $LEROBOT_DDP_STATIC_GRAPH"
echo "DDP gradient_as_bucket_view: $LEROBOT_DDP_GRADIENT_AS_BUCKET_VIEW"

if [[ "$RUN_FOREGROUND" == true ]]; then
    echo "============================================="
    set -o pipefail
    "${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
    exit ${PIPESTATUS[0]}
fi

nohup "${TRAIN_CMD[@]}" > "$LOG_FILE" 2>&1 &
PID=$!
echo "PID: $PID"
echo "============================================="
