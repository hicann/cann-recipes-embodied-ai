#!/bin/bash
#
# Copyright (c) 2026 Heng Fang (H-EmbodVis / DOMINO).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
set -euo pipefail

# Start the PUMA policy server on an Ascend NPU. This wraps the upstream
# DOMINO deployment entry (deployment/model_server/server_policy.py) with
# --device npu; the RoboTwin simulation side stays completely unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
DOMINO_ROOT="${DOMINO_ROOT:-$WORKSPACE_ROOT/DOMINO}"
PUMA_ROOT="$DOMINO_ROOT/policy/PUMA"
ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

CKPT_PATH=""
PORT=9001
NPU_ID="${NPU_ID:-0}"
EXTRA_ARGS=()

usage() {
    cat <<USAGE
Usage: $0 --ckpt <path> [OPTIONS] [-- EXTRA_SERVER_ARGS...]

Serve a PUMA checkpoint on an Ascend NPU.

Options:
  --ckpt PATH        Path to the PUMA checkpoint (required),
                     e.g. .../checkpoints/steps_100000_pytorch_model.pt
  --port PORT        Server port. Default: ${PORT}
  --npu ID           NPU device id (sets ASCEND_RT_VISIBLE_DEVICES). Default: ${NPU_ID}
  -h, --help         Show this help message.

Anything after "--" is forwarded to server_policy.py verbatim.

Examples:
  $0 --ckpt /data/ckpt/steps_100000_pytorch_model.pt
  $0 --ckpt /data/ckpt/steps_100000_pytorch_model.pt --port 9002 --npu 1
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)  CKPT_PATH="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --npu)   NPU_ID="$2"; shift 2 ;;
        --)      shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *)       echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$CKPT_PATH" ]]; then
    echo "[ERROR] --ckpt is required." >&2
    usage
    exit 1
fi
[[ -f "$CKPT_PATH" ]] || { echo "[ERROR] Checkpoint not found: $CKPT_PATH" >&2; exit 1; }

if [[ ! -d "$PUMA_ROOT" ]]; then
    echo "[ERROR] PUMA tree not found: $PUMA_ROOT" >&2
    echo "[ERROR] Run ./manipulation/puma/infer_with_torch/src/scripts/setup.sh first," >&2
    echo "[ERROR] or point DOMINO_ROOT at an existing DOMINO checkout." >&2
    exit 1
fi

if [[ -f "$ASCEND_SET_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$ASCEND_SET_ENV"
else
    echo "[WARN] CANN set_env.sh not found at $ASCEND_SET_ENV;" >&2
    echo "[WARN] assuming the CANN environment is already sourced." >&2
fi

cd "$PUMA_ROOT"
export PYTHONPATH="$PUMA_ROOT:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"

echo "[INFO] PUMA root:  $PUMA_ROOT"
echo "[INFO] Checkpoint: $CKPT_PATH"
echo "[INFO] Port:       $PORT"
echo "[INFO] NPU id:     $NPU_ID"
echo "[INFO] The first request is slower (device init, memory pool growth, weight staging); send one warm-up request before measuring latency."

exec python deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT_PATH" \
    --port "$PORT" \
    --device npu \
    --use_bf16 \
    "${EXTRA_ARGS[@]}"
