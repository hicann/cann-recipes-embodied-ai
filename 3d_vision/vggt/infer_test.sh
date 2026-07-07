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
# VGGT Inference Launch Script (YAML Configuration Mode)
#
# Usage:
#   bash infer_test.sh                    # Use default config single.yaml
#   bash infer_test.sh sp4.yaml           # Use specified config file
#
# Execution Flow:
#   infer_test.sh
#       ├─→ yaml_parse.sh (vggt_parse_config)  # Parse YAML configuration
#       └─→ vggt_launch.sh (vggt_launch)       # Launch inference task
# ============================================================

# CANN environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Get script directory
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Load YAML parsing script
source "${SCRIPT_PATH}/yaml_parse.sh"

# Load VGGT launch script
source "${SCRIPT_PATH}/vggt_launch.sh"

# Set YAML configuration file path
# Supports two ways to specify config file:
# 1. Command-line argument: bash infer_test.sh sp4.yaml
# 2. Default value: single.yaml
if [ -n "${1}" ]; then
    YAML_FILE_NAME="${1}"
else
    YAML_FILE_NAME="single.yaml"
fi

export YAML="${SCRIPT_PATH}/config/${YAML_FILE_NAME}"

# Execution flow: parse YAML first, then launch inference
vggt_parse_config   # Step 1: Parse YAML configuration
vggt_launch         # Step 2: Launch inference task