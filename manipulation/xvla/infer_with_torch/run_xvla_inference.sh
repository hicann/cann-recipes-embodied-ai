#!/bin/bash
#
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$WORKSPACE_ROOT/lerobot}"
DEFAULT_STORAGE_ROOT="$WORKSPACE_ROOT"
if [[ -d /data/docker/xvla_storage ]]; then
    DEFAULT_STORAGE_ROOT=/data/docker/xvla_storage
fi
XVLA_STORAGE_ROOT="${XVLA_STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"
export HOST_CACHE_CAPACITY="${HOST_CACHE_CAPACITY:-20}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$XVLA_STORAGE_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$XVLA_STORAGE_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$XVLA_STORAGE_ROOT/.cache/huggingface/transformers}"
export TOKENIZERS_PARALLELISM=false
export LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION="${LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION:-1}"
export LEROBOT_XVLA_USE_NPU_BADD_BMM="${LEROBOT_XVLA_USE_NPU_BADD_BMM:-1}"

if [[ ! -d "$LEROBOT_ROOT" ]]; then
    echo "LeRobot repo not found: $LEROBOT_ROOT"
    echo "Run: ./manipulation/xvla/train/src/scripts/setup.sh"
    exit 1
fi

cd "$LEROBOT_ROOT"
python "$SCRIPT_DIR/test_xvla_inference_ascend.py" "$@"
