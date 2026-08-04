#!/bin/bash
# coding=utf-8
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


# 检查传入参数数量
if [ $# -lt 1 ]; then
    echo "Usage: bash run_pi05_inference.sh [checkpoint] [dtype=float16] [warmup=10] [iters=100] [device=npu]"
    exit 1
fi

# 读取传入变量
CHECKPOINT=$1         # 第一个参数：模型路径/模型ID
DTYPE=${2:-float16}   # 第二个参数：数据类型，默认值 float16
WARMUP=${3:-10}       # 第三个参数：预热次数，默认值 10
ITERS=${4:-100}       # 第四个参数：迭代次数，默认值 100
DEVICE=${5:-npu}      # 第五个参数：设备，默认值 npu（可传 npu:1）

# 打印参数信息
echo "Running inference script with the following parameters:"
echo "Checkpoint: $CHECKPOINT"
echo "DTYPE: $DTYPE"
echo "Warmup: $WARMUP"
echo "Iterations: $ITERS"
echo "Device: $DEVICE"

# 运行 Python 示例脚本并保留真实退出状态
if python run_pi05_example.py \
    --pretrained_model_name_or_path "$CHECKPOINT" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --num_warmup "$WARMUP" \
    --num_inference "$ITERS"; then
    echo "脚本成功运行!"
else
    status=$?
    echo "脚本运行失败!"
    exit "$status"
fi
