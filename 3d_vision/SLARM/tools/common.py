# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


def is_ascend_npu():
    """Check if current environment is Ascend NPU"""
    try:
        # Try to import torch_npu
        import torch_npu
        # Check if NPU device is available
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        # torch_npu not installed or NPU not supported
        return False
