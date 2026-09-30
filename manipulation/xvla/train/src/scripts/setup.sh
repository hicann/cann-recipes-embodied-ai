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
XVLA_PATCH_PATH="$SCRIPT_DIR/../patches/lerobot_xvla_ascend.patch"
LEROBOT_COMMIT="3f2179f3b69708b6ad009b2e7685dd9d05269ee1"

CREATE_CONDA=false
ENV_NAME="lerobot-xvla"
PYTHON_VERSION="3.12"
TORCH_WHEEL=""
TORCHVISION_WHEEL=""
TORCH_NPU_WHEEL=""
SKIP_TORCH_CHECK=false

BASE_DEPS=(
    "datasets>=4.7.0,<5.0.0"
    "huggingface-hub[hf-transfer,cli]>=1.0.0,<2.0.0"
    "accelerate>=1.14.0,<2.0.0"
    "setuptools>=71.0.0,<81.0.0"
    "cmake>=3.29.0.1,<4.2.0"
    "einops>=0.8.0,<0.9.0"
    "opencv-python-headless>=4.9.0,<4.14.0"
    "av>=15.0.0,<16.0.0"
    "jsonlines>=4.0.0,<5.0.0"
    "packaging>=24.2,<26.0"
    "wandb>=0.24.0,<0.28.0"
    "draccus==0.10.0"
    "gymnasium>=1.1.1,<2.0.0"
    "termcolor>=2.4.0,<4.0.0"
    "tqdm>=4.66.0,<5.0.0"
    "safetensors>=0.4.3,<1.0.0"
    "hf-libero>=0.1.4,<0.2.0"
)

XVLA_DEPS=(
    "transformers>=5.4.0,<5.6.0"
)

usage() {
    cat <<USAGE
Usage: $0 [OPTIONS]

Prepare the XVLA training workspace on Ascend.

Options:
  --create-conda                  Create and activate a fresh conda env.
  --env-name NAME                 Conda env name when --create-conda is used. Default: ${ENV_NAME}
  --python-version VERSION        Python version when --create-conda is used. Default: ${PYTHON_VERSION}
  --torch-wheel PATH              Local path to a torch wheel to install.
  --torchvision-wheel PATH        Local path to a torchvision wheel to install.
  --torch-npu-wheel PATH          Local path to a torch_npu wheel to install.
  --skip-torch-check              Skip the final torch/torch_npu import check.
  -h, --help                      Show this help message.

Notes:
  1. By default the script uses the current active Python environment.
  2. Platform torch / torchvision / torch_npu packages are never downloaded automatically.
  3. If a shared Ascend environment is already available, activate it first and run this script without --create-conda.
USAGE
}

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "Required command not found: $1"
    fi
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

install_platform_stack() {
    local wheels=()
    if [[ -n "$TORCH_WHEEL" ]]; then [[ -f "$TORCH_WHEEL" ]] || error "torch wheel not found: $TORCH_WHEEL"; wheels+=("$TORCH_WHEEL"); fi
    if [[ -n "$TORCHVISION_WHEEL" ]]; then [[ -f "$TORCHVISION_WHEEL" ]] || error "torchvision wheel not found: $TORCHVISION_WHEEL"; wheels+=("$TORCHVISION_WHEEL"); fi
    if [[ -n "$TORCH_NPU_WHEEL" ]]; then [[ -f "$TORCH_NPU_WHEEL" ]] || error "torch_npu wheel not found: $TORCH_NPU_WHEEL"; wheels+=("$TORCH_NPU_WHEEL"); fi
    if (( ${#wheels[@]} > 0 )); then
        info "Installing platform torch stack from local wheels"
        pip install "${wheels[@]}"
    else
        info "No local torch wheels provided, reusing the current platform stack"
    fi
}

apply_patch_if_needed() {
    local patch_path="$1"
    [[ -f "$patch_path" ]] || error "Patch file not found: $patch_path"
    if git apply --check "$patch_path" >/dev/null 2>&1; then
        info "Applying patch: $(basename "$patch_path")"
        git apply "$patch_path"
    elif git apply --reverse --check "$patch_path" >/dev/null 2>&1; then
        info "Patch already applied: $(basename "$patch_path")"
    else
        error "Patch cannot be cleanly applied: $(basename "$patch_path"). Use a clean lerobot checkout at ${LEROBOT_COMMIT}."
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --create-conda) CREATE_CONDA=true; shift ;;
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --torch-wheel) TORCH_WHEEL="$2"; shift 2 ;;
        --torchvision-wheel) TORCHVISION_WHEEL="$2"; shift 2 ;;
        --torch-npu-wheel) TORCH_NPU_WHEEL="$2"; shift 2 ;;
        --skip-torch-check) SKIP_TORCH_CHECK=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

check_command git
if [[ "$CREATE_CONDA" == true ]]; then activate_conda_env; fi
check_command python
check_command pip

mkdir -p "$WORKSPACE_ROOT"
if [[ ! -d "$LEROBOT_ROOT/.git" ]]; then
    info "Cloning lerobot into $LEROBOT_ROOT"
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_ROOT"
else
    info "LeRobot repo already exists: $LEROBOT_ROOT"
fi

cd "$LEROBOT_ROOT"
git fetch origin "$LEROBOT_COMMIT" --depth=1 || true
git checkout "$LEROBOT_COMMIT"
apply_patch_if_needed "$XVLA_PATCH_PATH"

install_platform_stack

info "Installing LeRobot dependencies without changing the resolved torch stack"
pip install "${BASE_DEPS[@]}"
pip install "${XVLA_DEPS[@]}"
pip install -e . --no-deps

if [[ "$SKIP_TORCH_CHECK" == true ]]; then
    warn "Skipping torch / torch_npu validation by request"
elif python -c "import torch, torch_npu, transformers, safetensors; assert hasattr(torch, 'npu')" >/dev/null 2>&1; then
    info "Verified current environment can import torch, torch_npu, transformers and safetensors"
else
    cat <<MSG >&2
[ERROR] torch / torch_npu / transformers / safetensors is still unavailable after setup.

Prepare a compatible Ascend PyTorch environment first, or rerun setup.sh with local wheel paths:
  --torch-wheel /path/to/torch.whl
  --torchvision-wheel /path/to/torchvision.whl
  --torch-npu-wheel /path/to/torch_npu.whl
MSG
    exit 1
fi

cat <<MSG

=============================================
XVLA training workspace is ready.
LeRobot root: $LEROBOT_ROOT
LeRobot commit: $LEROBOT_COMMIT
XVLA patch: $XVLA_PATCH_PATH
=============================================
MSG
