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
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
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
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION="${LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION:-1}"
export LEROBOT_XVLA_USE_NPU_BADD_BMM="${LEROBOT_XVLA_USE_NPU_BADD_BMM:-1}"

infer_policy_device() {
    local previous=""
    local arg=""
    for arg in "$@"; do
        if [[ "$previous" == "--policy.device" ]]; then printf '%s' "$arg"; return 0; fi
        case "$arg" in --policy.device=*) printf '%s' "${arg#--policy.device=}"; return 0 ;; esac
        previous="$arg"
    done
    return 1
}

usage() {
    echo "Usage: $0 <lerobot-eval args>"
    echo "Example:"
    echo "  $0 --policy.path=../models/lerobot/xvla-libero --policy.device=npu --env.type=libero --env.task=libero_spatial --eval.n_episodes=10 --eval.batch_size=1 --output_dir=../evals/xvla_libero"
}

if [[ $# -eq 0 ]]; then usage; exit 0; fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
[[ -d "$LEROBOT_ROOT" ]] || { echo "LeRobot repo not found: $LEROBOT_ROOT"; echo "Run: ./manipulation/xvla/train/src/scripts/setup.sh"; exit 1; }
command -v lerobot-eval >/dev/null 2>&1 || { echo "lerobot-eval not found, please activate the proper environment."; exit 1; }

POLICY_DEVICE="$(infer_policy_device "$@" || true)"
if [[ -z "${LEROBOT_EVAL_NPU_DEVICE:-}" && "$POLICY_DEVICE" == npu* ]]; then
    export LEROBOT_EVAL_NPU_DEVICE="$POLICY_DEVICE"
fi

cd "$LEROBOT_ROOT"
"$(command -v lerobot-eval)" "$@"
