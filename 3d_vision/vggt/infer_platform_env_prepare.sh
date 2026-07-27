#!/usr/bin/env bash
set -euo pipefail

echo "========== 0. Check current directory =========="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}"

echo "[Info] Current script directory: ${SCRIPT_DIR}"

if [ "$(basename "${TARGET_DIR}")" != "vggt" ]; then
    echo "[Error] Please place the script in cann-recipes-embodied-ai/3d_vision/vggt directory to execute"
    exit 1
fi

echo "========== 1. Prepare VGGT official repository =========="

WORKSPACE_DIR="cann_recipes" # User should configure this
OFFICIAL_VGGT_DIR="${WORKSPACE_DIR}/vggt"

if [ ! -d "${OFFICIAL_VGGT_DIR}" ]; then
    git clone https://github.com/facebookresearch/vggt.git "${OFFICIAL_VGGT_DIR}"
else
    echo "[Skip] ${OFFICIAL_VGGT_DIR} already exists"
fi

echo "========== 2. Download VGGT model weights =========="

pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

if [ ! -f "${OFFICIAL_VGGT_DIR}/model.pt" ]; then
    # 'model.pt' required: without it, snapshot_download forces Xet/CAS protocol, incompatible with hf-mirror.com (401 Unauthorized).
    hf download facebook/VGGT-1B model.pt --local-dir "${OFFICIAL_VGGT_DIR}"
else
    echo "[Skip] ${OFFICIAL_VGGT_DIR}/model.pt already exists"
fi

echo "========== 3. Create ckpt directory and copy model weights =========="

CKPT_DIR="${TARGET_DIR}/ckpt"
mkdir -p "${CKPT_DIR}"

if [ -f "${OFFICIAL_VGGT_DIR}/model.pt" ]; then
    cp -n "${OFFICIAL_VGGT_DIR}/model.pt" "${CKPT_DIR}/model.pt"
    echo "[OK] model.pt copied to ${CKPT_DIR}/model.pt"
else
    echo "[Error] ${OFFICIAL_VGGT_DIR}/model.pt not found, please check if weights downloaded successfully"
    exit 1
fi

echo "========== 4. Copy VGGT network structure code to current project directory =========="

mkdir -p "${TARGET_DIR}/vggt"

cp -n "${OFFICIAL_VGGT_DIR}/visual_util.py" "${TARGET_DIR}/" || true
cp -rn "${OFFICIAL_VGGT_DIR}/examples" "${TARGET_DIR}/" || true

cp -rn "${OFFICIAL_VGGT_DIR}/vggt/dependency" "${TARGET_DIR}/vggt/dependency" || true
cp -rn "${OFFICIAL_VGGT_DIR}/vggt/heads" "${TARGET_DIR}/vggt/" || true
cp -rn "${OFFICIAL_VGGT_DIR}/vggt/layers" "${TARGET_DIR}/vggt/" || true
cp -rn "${OFFICIAL_VGGT_DIR}/vggt/utils" "${TARGET_DIR}/vggt/" || true

echo "========== 5. Install Python dependencies =========="

cd "${TARGET_DIR}"
pip3 install -r requirements.txt

echo "========== VGGT environment and code preparation completed =========="
echo "Project directory: ${TARGET_DIR}"
echo "Official VGGT repository directory: ${OFFICIAL_VGGT_DIR}"
echo "Weights file: ${CKPT_DIR}/model.pt"