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

# Prepare the PUMA Ascend inference workspace.
# Training and inference share the same pinned DOMINO commit and Ascend
# runtime, so this simply delegates to the train setup script and adds
# the RoboTwin evaluation communication dependencies on top.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SETUP="$SCRIPT_DIR/../../../train/src/scripts/setup.sh"

if [[ ! -f "$TRAIN_SETUP" ]]; then
    echo "[ERROR] Shared setup script not found: $TRAIN_SETUP" >&2
    exit 1
fi

exec bash "$TRAIN_SETUP" --with-eval-deps "$@"
