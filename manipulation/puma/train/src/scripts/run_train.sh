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

# Thin wrapper around the upstream DOMINO Ascend training launcher.
# All knobs are plain environment variables and are forwarded as-is; see
# policy/PUMA/scripts/run_scripts/run_lerobot_robotwin_puma_ascend.sh
# in the DOMINO repo for the full list.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
DOMINO_ROOT="${DOMINO_ROOT:-$WORKSPACE_ROOT/DOMINO}"
LAUNCHER="$DOMINO_ROOT/policy/PUMA/scripts/run_scripts/run_lerobot_robotwin_puma_ascend.sh"

usage() {
    cat <<USAGE
Usage: [ENV_VAR=...] $0

Launch PUMA 8-card DeepSpeed ZeRO-2 training on Ascend NPUs.

Required environment variables:
  DATA_ROOT_DIR                LeRobot-format DOMINO dataset root.

Commonly overridden environment variables (defaults in parentheses):
  DOMINO_ROOT                  DOMINO repo root ($WORKSPACE_ROOT/DOMINO)
  BASE_VLM                     Base VLM weights (playground/Pretrained_models/Qwen3-VL-4B-Instruct)
  NUM_GPUS                     Number of NPUs (8)
  ASCEND_RT_VISIBLE_DEVICES    Visible NPUs (0,1,2,3,4,5,6,7)
  ASCEND_SET_ENV               CANN set_env.sh path (/usr/local/Ascend/ascend-toolkit/set_env.sh)
  WORLD_MODEL_ENABLED          World-model supervision (true)
  PER_DEVICE_BATCH_SIZE        Per-NPU batch size (4)
  MAX_TRAIN_STEPS              Training steps (200000)
  ENABLE_WANDB                 Set 1 to log to Weights & Biases (0)
  RUN_ROOT_DIR                 Output root (results/Checkpoints)
  DRY_RUN                      Set 1 to preview the resolved command without launching.

Examples:
  DRY_RUN=1 DATA_ROOT_DIR=/path/to/lerobot_dataset $0
  DATA_ROOT_DIR=/path/to/lerobot_dataset $0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$LAUNCHER" ]]; then
    echo "[ERROR] DOMINO Ascend launcher not found: $LAUNCHER" >&2
    echo "[ERROR] Run ./manipulation/puma/train/src/scripts/setup.sh first," >&2
    echo "[ERROR] or point DOMINO_ROOT at an existing DOMINO checkout." >&2
    exit 1
fi

if [[ -z "${DATA_ROOT_DIR:-}" ]]; then
    echo "[ERROR] DATA_ROOT_DIR is required, e.g.:" >&2
    echo "[ERROR]   DATA_ROOT_DIR=/path/to/lerobot_dataset $0" >&2
    exit 1
fi

echo "[INFO] DOMINO root: $DOMINO_ROOT"
echo "[INFO] Launcher:    $LAUNCHER"
echo "[INFO] Dataset:     $DATA_ROOT_DIR"

exec bash "$LAUNCHER"
