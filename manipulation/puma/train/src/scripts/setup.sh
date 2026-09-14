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

# Prepare the PUMA (DOMINO) Ascend training workspace:
# clone the upstream DOMINO repo, pin it to the verified commit and
# install the pinned Ascend runtime. Ascend adaptation code already
# lives in upstream DOMINO, so no patch is applied here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
DOMINO_ROOT="${DOMINO_ROOT:-$WORKSPACE_ROOT/DOMINO}"
DOMINO_REPO_URL="${DOMINO_REPO_URL:-https://github.com/H-EmbodVis/DOMINO.git}"
DOMINO_COMMIT="9c94f2d3a700fff3b65f041df4038d497139ed1f"  # 2026-08-18, Ascend train + infer support

CREATE_CONDA=false
ENV_NAME="puma-ascend"
PYTHON_VERSION="3.10"
WITH_EVAL_DEPS=false
SKIP_TORCH_CHECK=false

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS]

Prepare the PUMA training workspace on Ascend.

Options:
  --create-conda              Create and activate a fresh conda env.
  --env-name NAME             Conda env name when --create-conda is used. Default: ${ENV_NAME}
  --python-version VERSION    Python version when --create-conda is used. Default: ${PYTHON_VERSION}
  --with-eval-deps            Also install RoboTwin evaluation communication dependencies.
  --skip-torch-check          Skip the final torch/torch_npu import check.
  -h, --help                  Show this help message.

Notes:
  1. By default the script uses the current active Python environment.
  2. requirements-ascend.txt pins torch==2.5.1 / torch-npu==2.5.1.post1 and is
     installed FIRST so later dependencies cannot pull GPU-only wheels.
  3. Source your CANN toolkit env before training, e.g.:
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
USAGE
}

info()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

check_command() {
    command -v "$1" >/dev/null 2>&1 || error "Required command not found: $1"
}

activate_conda_env() {
    check_command conda
    eval "$(conda shell.bash hook)"
    if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        info "Conda env already exists: $ENV_NAME"
    else
        info "Creating conda env: $ENV_NAME (python=${PYTHON_VERSION})"
        conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
    fi
    conda activate "$ENV_NAME"
    info "Activated conda env: $ENV_NAME"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --create-conda)   CREATE_CONDA=true; shift ;;
        --env-name)       ENV_NAME="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --with-eval-deps) WITH_EVAL_DEPS=true; shift ;;
        --skip-torch-check) SKIP_TORCH_CHECK=true; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                error "Unknown argument: $1" ;;
    esac
done

check_command git

if [[ "$CREATE_CONDA" == true ]]; then
    activate_conda_env
fi

check_command python
check_command pip

mkdir -p "$WORKSPACE_ROOT"

if [[ ! -d "$DOMINO_ROOT/.git" ]]; then
    info "Cloning DOMINO into $DOMINO_ROOT"
    git clone "$DOMINO_REPO_URL" "$DOMINO_ROOT"
else
    info "DOMINO repo already exists: $DOMINO_ROOT"
fi

cd "$DOMINO_ROOT"
git fetch origin "$DOMINO_COMMIT" --depth=1 2>/dev/null || git fetch origin || true
git checkout "$DOMINO_COMMIT"
info "DOMINO pinned to commit $DOMINO_COMMIT"

cd "$DOMINO_ROOT/policy/PUMA"

info "Installing pinned Ascend runtime (requirements-ascend.txt) first"
pip install -r requirements-ascend.txt

info "Installing PUMA in editable mode (no build isolation)"
pip install --no-build-isolation -e .

if [[ "$WITH_EVAL_DEPS" == true ]]; then
    info "Installing RoboTwin evaluation communication dependencies"
    pip install -r examples/Robotwin/eval_files/requirements.txt
fi

if [[ "$SKIP_TORCH_CHECK" == true ]]; then
    warn "Skipping torch / torch_npu validation by request"
elif python -c "import torch, torch_npu, transformers, deepspeed" >/dev/null 2>&1; then
    info "Verified current environment can import torch, torch_npu, transformers and deepspeed"
else
    cat <<'MSG' >&2
[ERROR] torch / torch_npu / transformers / deepspeed is still unavailable after setup.

Check that:
  1. The CANN toolkit env is sourced:
       source /usr/local/Ascend/ascend-toolkit/set_env.sh
  2. Your CANN version matches the torch-npu==2.5.1.post1 requirement
     (see https://gitee.com/ascend/pytorch for the compatibility table).
  3. numpy stayed at 1.26.4 (later installs may silently upgrade it to 2.x,
     which breaks torch-npu; re-pin it if that happened).
MSG
    exit 1
fi

cat <<MSG

=============================================
PUMA Ascend training workspace is ready.
DOMINO root:   $DOMINO_ROOT
DOMINO commit: $DOMINO_COMMIT

Recommended next steps:
  1. Put the base VLM under:
       $DOMINO_ROOT/policy/PUMA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
  2. Put Grounded-SAM-2 weights under:
       $DOMINO_ROOT/policy/PUMA/playground/Pretrained_models/grounded_sam2
  3. Prepare the LeRobot-format DOMINO dataset and copy modality.json
     into each task's meta/ folder.
  4. Dry-run first:
       DRY_RUN=1 DATA_ROOT_DIR=/path/to/lerobot_dataset \\
         ./manipulation/puma/train/src/scripts/run_train.sh
=============================================
MSG
