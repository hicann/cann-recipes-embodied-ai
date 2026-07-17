#!/bin/bash
#
# Copyright (c) 2026 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 Shanghai Jiao Tong University
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RECIPE_REPO_ROOT/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$WORKSPACE_ROOT/lerobot}"
PATCH_PATH="$SCRIPT_DIR/../patches/lerobot_ascend_train_common.patch"
LEROBOT_COMMIT="58f70b6bd370864139a3795ac3497a9eae8c42d5"

CREATE_CONDA=false
ENV_NAME="lerobot-act"
PYTHON_VERSION="3.10"
TORCH_WHEEL=""
TORCHVISION_WHEEL=""
TORCH_NPU_WHEEL=""
SKIP_TORCH_CHECK=false

BASE_DEPS=(
    "datasets>=4.0.0,<4.2.0"
    "diffusers>=0.27.2,<0.36.0"
    "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0"
    "accelerate>=1.10.0,<2.0.0"
    "setuptools>=71.0.0,<81.0.0"
    "cmake>=3.29.0.1,<4.2.0"
    "einops>=0.8.0,<0.9.0"
    "opencv-python-headless>=4.9.0,<4.13.0"
    "av>=15.0.0,<16.0.0"
    "jsonlines>=4.0.0,<5.0.0"
    "packaging>=24.2,<26.0"
    "pynput>=1.7.7,<1.9.0"
    "pyserial>=3.5,<4.0"
    "wandb>=0.20.0,<0.22.0"
    "draccus==0.10.0"
    "gymnasium>=1.1.1,<2.0.0"
    "rerun-sdk>=0.24.0,<0.27.0"
    "deepdiff>=7.0.1,<9.0.0"
    "imageio[ffmpeg]>=2.34.0,<3.0.0"
    "termcolor>=2.4.0,<4.0.0"
    "gym-aloha>=0.1.2,<0.2.0"
)

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Prepare the ACT training workspace on Ascend.

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
  2. The script installs LeRobot common Python dependencies and ALOHA dependencies.
  3. Platform-specific torch / torchvision / torch_npu packages are not hard-coded.
     Either prepare them in the current env in advance, or pass local wheel paths above.
EOF
}

info() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARN] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "Required command not found: $1"
    fi
}

activate_conda_env() {
    check_command conda
    # shellcheck disable=SC1091
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

    if [[ -n "$TORCH_WHEEL" ]]; then
        [[ -f "$TORCH_WHEEL" ]] || error "torch wheel not found: $TORCH_WHEEL"
        wheels+=("$TORCH_WHEEL")
    fi
    if [[ -n "$TORCHVISION_WHEEL" ]]; then
        [[ -f "$TORCHVISION_WHEEL" ]] || error "torchvision wheel not found: $TORCHVISION_WHEEL"
        wheels+=("$TORCHVISION_WHEEL")
    fi
    if [[ -n "$TORCH_NPU_WHEEL" ]]; then
        [[ -f "$TORCH_NPU_WHEEL" ]] || error "torch_npu wheel not found: $TORCH_NPU_WHEEL"
        wheels+=("$TORCH_NPU_WHEEL")
    fi

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
        error "Patch cannot be cleanly applied: $(basename "$patch_path"). Please reset local lerobot changes or prepare a clean checkout at commit ${LEROBOT_COMMIT}."
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --create-conda)
            CREATE_CONDA=true
            shift
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --torch-wheel)
            TORCH_WHEEL="$2"
            shift 2
            ;;
        --torchvision-wheel)
            TORCHVISION_WHEEL="$2"
            shift 2
            ;;
        --torch-npu-wheel)
            TORCH_NPU_WHEEL="$2"
            shift 2
            ;;
        --skip-torch-check)
            SKIP_TORCH_CHECK=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            ;;
    esac
done

check_command git

if [[ "$CREATE_CONDA" == true ]]; then
    activate_conda_env
fi

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

apply_patch_if_needed "$PATCH_PATH"

install_platform_stack

info "Installing ACT training Python dependencies without changing the resolved torch stack"
pip install "${BASE_DEPS[@]}"
pip install -e . --no-deps

if [[ "$SKIP_TORCH_CHECK" == true ]]; then
    warn "Skipping torch / torch_npu validation by request"
elif python -c "import torch, torch_npu" >/dev/null 2>&1; then
    info "Verified current environment can import torch and torch_npu"
else
    cat <<MSG >&2
[ERROR] torch / torch_npu is still unavailable after setup.

This recipe does not hard-code a torch_npu download URL because the valid wheel set
depends on your Ascend software stack, architecture and CANN version.

You have two supported options:
  1. Activate a prebuilt Ascend training environment first, then rerun setup.sh.
  2. Rerun setup.sh and pass local wheel paths:
     --torch-wheel /path/to/torch.whl
     --torchvision-wheel /path/to/torchvision.whl
     --torch-npu-wheel /path/to/torch_npu.whl
MSG
    exit 1
fi

cat <<MSG

[SUCCESS] ACT training workspace is prepared.

LeRobot root:
  $LEROBOT_ROOT

Pinned commit:
  $LEROBOT_COMMIT

Patch file:
  $PATCH_PATH

Python environment:
  $(python -V 2>&1)
  Python executable: $(command -v python)

Next steps:
  1. Ensure the ALOHA dataset root exists.
  2. Ensure resnet18-f37072fd.pth is cached if the machine has no internet.
  3. Run:
     ./manipulation/act/train/src/scripts/run_train.sh act_aloha_smoke --port 29510
MSG
