# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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

#!/bin/bash

# ============================================================
# VGGT Inference Launch Script
# Function: Set environment variables and launch inference tasks
# Prerequisites: Must call yaml_parse.sh to parse configuration first (exports WORLD_SIZE, MASTER_PORT, ENTRY_SCRIPT, MODEL_ARGS)
# ============================================================

# Set environment variables (NPU optimization)
function vggt_setup_env()
{
    # NPU optimization environment variables
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-"expandable_segments:True"}
    export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-"2"}
    export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-"1"}
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}

    # HCCL distributed communication configuration
    LOCAL_HOST=$(hostname -I | awk -F " " '{print$1}')
    export HCCL_IF_IP=${LOCAL_HOST}
    export HCCL_IF_BASE_PORT=23456
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=1200

    echo "[INFO] NPU environment configured"
}

# Launch inference task
function vggt_launch_task()
{
    # Check if required environment variables are set (exported by yaml_parse.sh)
    if [ -z "${WORLD_SIZE}" ]; then
        echo "[ERROR] WORLD_SIZE is not set. Please run vggt_parse_config first."
        exit 1
    fi
    if [ -z "${ENTRY_SCRIPT}" ]; then
        echo "[ERROR] ENTRY_SCRIPT is not set. Please run vggt_parse_config first."
        exit 1
    fi

    echo "[INFO] Entry script: ${ENTRY_SCRIPT}"
    echo "[INFO] World size: ${WORLD_SIZE}"
    echo "[INFO] Master port: ${MASTER_PORT}"
    echo "[INFO] Config file: ${YAML}"
    echo "==================================>"

    if command -v torchrun >/dev/null 2>&1; then
        echo "[INFO] Launching with torchrun (nproc_per_node=${WORLD_SIZE}, master_port=${MASTER_PORT})"
        torchrun --master_port=${MASTER_PORT} \
                 --nproc_per_node=${WORLD_SIZE} \
                 ${ENTRY_SCRIPT} \
                 --config ${YAML}
    else
        echo "[INFO] torchrun not found, using python -m torch.distributed.run"
        python -m torch.distributed.run --master_port=${MASTER_PORT} \
                 --nproc_per_node=${WORLD_SIZE} \
                 ${ENTRY_SCRIPT} \
                 --config ${YAML}
    fi
}

# VGGT launch main entry point
function vggt_launch()
{
    vggt_setup_env
    vggt_launch_task
}