#!/bin/bash
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
#
# Adapt Cosmos3 source tree for Ascend NPU.
#
# This script contains only simple/mechanical edits from:
#   a61b292..a93d52c
# It intentionally excludes uv.lock and complex feature patches such as:
#   - local checkpoint/tokenizer loading changes
#   - Qwen3-VL CPU preprocessing workaround
#
# Run from the Cosmos3 repository root:
#   bash npu_adapt.sh

set -euo pipefail

COSMOS_ROOT="${COSMOS_ROOT:-.}"
EXPECTED_COSMOS_COMMIT="${EXPECTED_COSMOS_COMMIT:-a61b292}"

info() {
    printf '\033[32m[INFO]\033[0m %s\n' "$*"
}

warn() {
    printf '\033[33m[WARN]\033[0m %s\n' "$*"
}

file_path() {
    printf '%s/%s' "$COSMOS_ROOT" "$1"
}

repo_file() {
    printf '%s/%s' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$1"
}

ensure_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        warn "Skip missing file: $file"
        return 1
    fi
    return 0
}

verify_expected_commit() {
    local head

    if ! git -C "$COSMOS_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf '[ERROR] %s is not a git work tree; refusing to run source rewrites.\n' "$COSMOS_ROOT" >&2
        printf '[ERROR] Run this script from the cosmos-framework repository checked out at %s.\n' "$EXPECTED_COSMOS_COMMIT" >&2
        exit 1
    fi

    head="$(git -C "$COSMOS_ROOT" rev-parse --short=7 HEAD)"
    if [ "$head" != "$EXPECTED_COSMOS_COMMIT" ]; then
        printf '[ERROR] Expected cosmos-framework HEAD %s, got %s.\n' "$EXPECTED_COSMOS_COMMIT" "$head" >&2
        printf '[ERROR] Refusing to run because sed/perl rewrites are tied to the expected upstream source layout.\n' >&2
        exit 1
    fi

    info "Verified cosmos-framework base commit: $head"
}

delete_line_matching() {
    local file="$1"
    local pattern="$2"
    ensure_file "$file" || return 0
    sed -i "/${pattern}/d" "$file"
}

ensure_inference_default_npu() {
    local file
    file="$(file_path "cosmos_framework/scripts/inference.py")"
    ensure_file "$file" || return 0

    if ! grep -q '^import torch_npu' "$file"; then
        sed -i '4i import torch\nimport torch_npu\ntorch.set_default_device("npu")\n' "$file"
        info "Added torch_npu default device setup to $file"
    else
        info "torch_npu default device setup already exists in $file"
    fi
}

adapt_flags() {
    local file
    file="$(file_path "cosmos_framework/utils/flags.py")"
    ensure_file "$file" || return 0

    sed -i 's|TRAINING: Final\[bool\] = _get_bool("COSMOS_TRAINING", True)|TRAINING: Final[bool] = _get_bool("COSMOS_TRAINING", False)|' "$file"
    sed -i 's|DEVICE: Final\[Device\] = Device(os.environ.get("COSMOS_DEVICE", "cuda").lower())|DEVICE: Final[Device] = Device(os.environ.get("COSMOS_DEVICE", "npu").lower())|' "$file"

    if ! grep -q '^[[:space:]]*NPU = "npu"' "$file"; then
        sed -i '/^[[:space:]]*META = "meta"/a\    NPU = "npu"' "$file"
        info "Added Device.NPU to $file"
    else
        info "Device.NPU already exists in $file"
    fi
}

adapt_model_loader() {
    local file
    file="$(file_path "cosmos_framework/utils/vfm/model_loader.py")"
    ensure_file "$file" || return 0

    if ! grep -q 'backend.endswith("hccl")' "$file"; then
        sed -i '/backend.endswith("nccl")/{n; a\    elif backend.endswith("hccl"):\n        return torch.device("npu", torch.npu.current_device())
}' "$file"
        info "Added HCCL backend device mapping to $file"
    else
        info "HCCL backend device mapping already exists in $file"
    fi
}

adapt_device_memory() {
    local file
    file="$(file_path "cosmos_framework/inference/args.py")"
    ensure_file "$file" || return 0

    sed -i '/^import pynvml$/d' "$file"

    # Replace the NVML/CUDA memory probe with an NPU memory probe. This pattern
    # targets the upstream implementation around _get_device_memory_bytes().
    perl -0pi -e 's/def _get_device_memory_bytes\(\) -> int:\n    try:\n        pynvml\.nvmlInit\(\)\n        handle = pynvml\.nvmlDeviceGetHandleByIndex\(0\)\n        info = pynvml\.nvmlDeviceGetMemoryInfo\(handle\)\n        pynvml\.nvmlShutdown\(\)\n        return info\.total\n    except Exception:\n        # Fallback for unified memory architectures \(e\.g\., GB10\) where\n        # nvmlDeviceGetMemoryInfo is not supported\.\n        import torch\n        if torch\.cuda\.is_available\(\):\n            return int\(torch\.cuda\.get_device_properties\(0\)\.total_memory\)\n        return 128 \* 1024\*\*3  # Default 128GB\n/def _get_device_memory_bytes() -> int:\n    try:\n        import torch\n        if torch.npu.is_available():\n            return int(torch.npu.get_device_properties(0).total_memory)\n        return 64 * 1024**3\n    except Exception:\n        return 64 * 1024**3\n/s' "$file"

    info "Adapted device memory probe in $file"
}

adapt_compile_defaults() {
    local file
    file="$(file_path "cosmos_framework/inference/common/args.py")"
    ensure_file "$file" || return 0

    sed -i 's|use_torch_compile: bool = True|use_torch_compile: bool = False|' "$file"
    sed -i 's|use_cuda_graphs: bool = True|use_cuda_graphs: bool = False|' "$file"
    info "Disabled compile/cuda-graphs defaults in $file"
}

adapt_common_init() {
    local file
    file="$(file_path "cosmos_framework/inference/common/init.py")"
    ensure_file "$file" || return 0

    sed -i 's|# 1. torch.cuda.set_device(local_rank) runs before any CUDA allocations,|# 1. torch.npu.set_device(local_rank) runs before any device allocations,|' "$file"
    sed -i 's|#    ensuring each rank places tensors on its own GPU (not all on cuda:0).|#    ensuring each rank places tensors on its own NPU (not all on npu:0).|' "$file"
    sed -i 's|if torch.cuda.is_available():|if torch.npu.is_available():|' "$file"
    sed -i 's|torch.cuda.set_per_process_memory_fraction(device_memory_fraction)|torch.npu.set_per_process_memory_fraction(device_memory_fraction)|' "$file"
    info "Adapted common init CUDA probes to NPU in $file"
}

adapt_common_inference() {
    local file
    file="$(file_path "cosmos_framework/inference/common/inference.py")"
    ensure_file "$file" || return 0

    sed -i 's|error_flag = torch.zeros(1, dtype=torch.int32, device="cuda")  # \[1\]|error_flag = torch.zeros(1, dtype=torch.int32, device=torch.device("npu"))|' "$file"
    info "Adapted distributed error flag device in $file"
}

adapt_distributed_runtime() {
    local file
    file="$(file_path "cosmos_framework/utils/distributed.py")"
    ensure_file "$file" || return 0

    sed -i 's|^import pynvml$|try:\n    import pynvml\nexcept ImportError:\n    pynvml = None|' "$file"
    sed -z -i 's|    if dist\.is_initialized():\n        return torch\.cuda\.current_device()|    if dist.is_initialized():\n        return torch.npu.current_device()|' "$file"
    sed -z -i 's|    # Set GPU affinity\.\n    pynvml\.nvmlInit()\n    local_rank = int(os\.getenv("LOCAL_RANK", 0))\n    try:\n        device = Device(local_rank)\n        os\.sched_setaffinity(0, device\.get_cpu_affinity())\n    except pynvml\.NVMLError as e:\n        log\.warning(f"Failed to set device affinity: {e}")|    # Set GPU affinity when NVML is available.\n    local_rank = int(os.getenv("LOCAL_RANK", 0))\n    if pynvml is not None:\n        try:\n            pynvml.nvmlInit()\n            device = Device(local_rank)\n            os.sched_setaffinity(0, device.get_cpu_affinity())\n        except pynvml.NVMLError as e:\n            log.warning(f"Failed to set device affinity: {e}")|' "$file"

    sed -i 's|# Set up NCCL communication\.|# Set up HCCL communication.|' "$file"
    sed -i 's|torch\.cuda\.set_device(local_rank)|torch.npu.set_device(local_rank)|' "$file"
    sed -i 's|dist\.init_process_group(backend="nccl"|dist.init_process_group(backend="hccl"|' "$file"
    sed -i 's|Initialized distributed training|Initialized distributed runtime|' "$file"
    sed -i 's|Training with {get_world_size()} GPUs\.|Running with {get_world_size()} NPUs.|' "$file"
    sed -i 's|return torch\.cuda\.current_device() == 0|return torch.npu.current_device() == 0|' "$file"

    info "Adapted distributed runtime to HCCL/NPU in $file"
}

adapt_inference_runtime() {
    local file
    file="$(file_path "cosmos_framework/inference/inference.py")"
    ensure_file "$file" || return 0

    sed -i 's|device: Any = "cuda"|device: Any = "npu"|' "$file"
    sed -i 's|torch.cuda.Stream|torch.npu.Stream|g' "$file"
    sed -i 's|torch.cuda.device_count()|torch.npu.device_count()|g' "$file"
    sed -i 's|torch.device("cuda", vae_device_index)|torch.device("npu", vae_device_index)|' "$file"
    sed -i 's|torch.device("cuda", torch.cuda.current_device())|torch.device("npu", torch.npu.current_device())|' "$file"
    sed -i 's|torch.cuda.Event()|torch.npu.Event()|' "$file"
    sed -i 's|torch.cuda.current_stream|torch.npu.current_stream|g' "$file"
    sed -i 's|torch.cuda.stream|torch.npu.stream|' "$file"
    sed -i 's|count_tensor = torch.tensor(\[num_local_batches\], dtype=torch.long, device="cuda")|count_tensor = torch.tensor([num_local_batches], dtype=torch.long, device="npu")|' "$file"
    info "Adapted inference runtime CUDA APIs to NPU in $file"
}

adapt_transfer_and_vision_inputs() {
    local file

    file="$(file_path "cosmos_framework/inference/transfer.py")"
    ensure_file "$file" || return 0
    sed -i 's|\.cuda()|.npu()|g' "$file"
    sed -i 's|device="cuda"|device="npu"|' "$file"

    file="$(file_path "cosmos_framework/inference/vision.py")"
    ensure_file "$file" || return 0
    sed -i 's|\.cuda()|.npu()|g' "$file"

    info "Adapted transfer/vision input tensor placement to NPU"
}

adapt_sequence_packing_device_move() {
    local file
    file="$(file_path "cosmos_framework/data/vfm/sequence_packing.py")"
    ensure_file "$file" || return 0

    sed -i 's|def to_cuda(self) -> None:|def to_npu(self) -> None:|g' "$file"
    sed -i 's|Move all tensor fields to CUDA in-place.|Move all tensor fields to NPU in-place.|g' "$file"
    sed -i 's|\.cuda()|.npu()|g' "$file"
    sed -i 's|\.to_cuda()|.to_npu()|g' "$file"
    info "Adapted PackedSequence device move helpers to NPU in $file"
}

adapt_omni_device_condition() {
    local file
    file="$(file_path "cosmos_framework/model/vfm/omni_mot_model.py")"
    ensure_file "$file" || return 0

    sed -i 's|if DEVICE == Device.CUDA:|if DEVICE in (Device.CUDA, Device.NPU):|' "$file"
    sed -i 's|torch.cuda.empty_cache()|torch.npu.empty_cache()|' "$file"
    sed -i 's|packed_sequence.to_cuda()|packed_sequence.to_npu()|g' "$file"
    info "Allowed OmniMoT initialization on Device.NPU in $file"
}

adapt_moe_triton_import_guard() {
    local file
    file="$(file_path "cosmos_framework/model/vfm/vlm/qwen3_vl_moe/moe_kernels.py")"
    ensure_file "$file" || return 0

    if grep -q '_HAS_TRITON' "$file"; then
        info "MoE Triton import guard already exists in $file"
        return 0
    fi

    perl -0pi -e 's/import torch\nimport triton\nimport triton\.language as tl/import torch\n\ntry:\n    import triton\n    import triton.language as tl\n    _HAS_TRITON = True\nexcept ImportError:\n    triton = None\n    tl = None\n    _HAS_TRITON = False/s' "$file"
    perl -0pi -e 's/\n\@triton\.jit\n(def _fill_indices_kernel\(.*?\n)(?=\ndef _fill_indices_wrapper\()/my $block = "\@triton.jit\n$1"; $block =~ s|^(?=.)|    |mg; "\nif _HAS_TRITON:\n\n$block\nelse:\n    _fill_indices_kernel = None\n\n"/se' "$file"
    info "Added MoE Triton import guard to $file"
}

adapt_attention_cann_registration() {
    local cann_init backends frontend arch_utils
    cann_init="$(file_path "cosmos_framework/model/attention/cann/__init__.py")"
    backends="$(file_path "cosmos_framework/model/attention/backends.py")"
    frontend="$(file_path "cosmos_framework/model/attention/frontend.py")"
    arch_utils="$(file_path "cosmos_framework/model/attention/utils/__init__.py")"

    ensure_file "$cann_init" || return 0
    ensure_file "$backends" || return 0
    ensure_file "$frontend" || return 0
    ensure_file "$arch_utils" || return 0

    if ! grep -q 'from cosmos_framework.model.attention.cann.checks import cann_attention_check' "$backends"; then
        sed -i '/from cosmos_framework.model.attention.flash2.checks import flash2_attention_check/i from cosmos_framework.model.attention.cann.checks import cann_attention_check' "$backends"
    fi
    if ! grep -q '"cann": cann_attention_check' "$backends"; then
        sed -i '/"flash3": flash3_attention_check,/a\    "cann": cann_attention_check,' "$backends"
    fi
    if ! grep -q 'device.type == "npu"' "$backends"; then
        perl -0pi -e 's/    arch_tag = get_arch_tag\(device\)\n    backend_list = get_backend_list\(arch_tag\)/    if device.type == "npu":\n        backend_list = ["cann"]\n    else:\n        arch_tag = get_arch_tag(device)\n        backend_list = get_backend_list(arch_tag)/s' "$backends"
    fi
    perl -0pi -e 's/backend_list = filter_attention_backends\(\["cann"\]\)/backend_list = ["cann"]/g' "$backends"
    perl -0pi -e 's/(elif arch_tag >= 80:\n        default_backends = \[\n            "flash2",\n            "natten",\n)            "cann",\n/$1/s' "$backends"
    perl -0pi -e 's/default_backends = \["natten", "cann"\]/default_backends = ["natten"]/g' "$backends"
    perl -0pi -e 's/\n        print\("80\+ arch detected, enabling CANN backend for NPU support\."\)//g' "$backends"
    perl -0pi -e 's/\n        print\(f"Unsupported arch tag \{arch_tag\} for Attention backends\. Returning empty backend list\."\)//g' "$backends"

    if ! grep -q 'from cosmos_framework.model.attention.cann import cann_attention' "$frontend"; then
        sed -i '/from cosmos_framework.model.attention.natten import natten_attention, natten_multi_dim_attention/a from cosmos_framework.model.attention.cann import cann_attention' "$frontend"
    fi
    if ! grep -q '"cann": cann_attention' "$frontend"; then
        sed -i '/"flash3": flash3_attention,/a\    "cann": cann_attention,' "$frontend"
    fi

    perl -0pi -e 's/    if hasattr\(torch, .npu.\) and torch\.npu\.is_available\(\) and \(device is None or device\.type == "npu"\):\n        return 80\n//' "$arch_utils"

    info "Registered CANN FIA attention backend"
}

apply_inference_local_checkpoint_patch() {
    local marker_file patch_file
    marker_file="$(file_path "cosmos_framework/inference/inference.py")"
    patch_file="$(repo_file "adaptor_patches/inference_local_checkpoint.patch")"

    ensure_file "$marker_file" || return 0
    ensure_file "$patch_file" || return 0

    if grep -q 'Path(checkpoint_path, "text_tokenizer").is_dir()' "$marker_file"; then
        info "Local checkpoint resource loading patch already applied in $marker_file"
        return 0
    fi

    patch -d "$COSMOS_ROOT" -p1 -i "$patch_file"
    info "Applied local checkpoint resource loading patch"
}

adapt_optional_nvml() {
    local file
    file="$(file_path "cosmos_framework/utils/device.py")"
    ensure_file "$file" || return 0

    sed -i 's|^import pynvml$|try:\n    import pynvml\nexcept ImportError:\n    pynvml = None|' "$file"
    sed -i 's|except pynvml.NVMLError as error:|except Exception as error:|g' "$file"
    sed -z -i 's|finally:\n        pynvml\.nvmlShutdown()|finally:\n        if pynvml is not None:\n            pynvml.nvmlShutdown()|' "$file"

    info "Made pynvml optional in device utilities"
}

main() {
    verify_expected_commit

    ensure_inference_default_npu
    adapt_flags
    adapt_model_loader
    adapt_device_memory
    adapt_compile_defaults
    adapt_common_init
    adapt_common_inference
    adapt_distributed_runtime
    adapt_inference_runtime
    adapt_transfer_and_vision_inputs
    adapt_sequence_packing_device_move
    adapt_omni_device_condition
    adapt_moe_triton_import_guard
    adapt_attention_cann_registration
    apply_inference_local_checkpoint_patch
    adapt_optional_nvml

    info "Simple NPU adaptation completed."
}

main "$@"
