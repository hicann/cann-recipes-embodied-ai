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
# YAML Configuration Parsing Script
# Function: Parse YAML configuration file, export environment variables, build command-line arguments
# Note: This script only handles parsing, not launching inference tasks
# ============================================================

# Check if YAML file exists
function vggt_check_yaml()
{
    if [ -z "${YAML}" ]; then
        echo "[ERROR] YAML is not set. Please export YAML before calling vggt_parse_config."
        exit 1
    fi
    if [ ! -f "${YAML}" ]; then
        echo "[ERROR] YAML file not found: ${YAML}"
        exit 1
    fi
    echo "[INFO] Using YAML config: ${YAML}"
}

# Validate YAML structure (required fields)
function vggt_validate_yaml()
{
    python3 - <<'PY_VALIDATE'
import os, sys, yaml
try:
    with open(os.environ["YAML"], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"[ERROR] Failed to parse YAML {os.environ['YAML']}: {e}")
    sys.exit(1)

REQUIRED_TOP = {"model_name", "world_size", "entry_script"}
missing = REQUIRED_TOP - set(cfg.keys())
if missing:
    print(f"[ERROR] Missing required top-level YAML key(s): {sorted(missing)}")
    sys.exit(1)

ws = cfg.get("world_size")
if not isinstance(ws, int) or ws <= 0:
    print(f"[ERROR] world_size must be a positive int, got {ws!r}")
    sys.exit(1)

print("[INFO] YAML validation passed.")
PY_VALIDATE
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

# Parse YAML metadata and export environment variables
# Security: Use environment variable to pass YAML path, avoiding shell injection
function vggt_parse_yaml()
{
    export YAML_PATH="${YAML}"
    export MODEL_NAME=$(python3 -c "import os, yaml; print(yaml.safe_load(open(os.environ['YAML_PATH']))['model_name'])")
    export WORLD_SIZE=$(python3 -c "import os, yaml; print(yaml.safe_load(open(os.environ['YAML_PATH']))['world_size'])")
    export MASTER_PORT=$(python3 -c "import os, yaml; print(yaml.safe_load(open(os.environ['YAML_PATH'])).get('master_port', 29600))")
    export ENTRY_SCRIPT=$(python3 -c "import os, yaml; print(yaml.safe_load(open(os.environ['YAML_PATH']))['entry_script'])")

    echo "[INFO] model_name: ${MODEL_NAME}"
    echo "[INFO] world_size: ${WORLD_SIZE}"
    echo "[INFO] master_port: ${MASTER_PORT}"
    echo "[INFO] entry_script: ${ENTRY_SCRIPT}"
}

# YAML parsing main entry point (only parsing, no launching)
function vggt_parse_config()
{
    vggt_check_yaml
    vggt_validate_yaml
    vggt_parse_yaml
    echo "[INFO] YAML parsing completed"
}