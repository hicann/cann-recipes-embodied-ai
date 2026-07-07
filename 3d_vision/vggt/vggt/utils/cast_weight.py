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
import torch
import torch_npu


def cast_model_weight(model):
    def __format_cast(module, class_name):
        if issubclass(class_name, torch.nn.Conv2d):
            if module.groups > 1:
                return module
            if hasattr(module, "weight") and module.weight is not None and \
                "weight" in dict(module.named_parameters()):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 4)
        return module

    def cast_weight(module):
        current_class = module.__class__
        module = __format_cast(module, current_class)
        if not list(module.children()):
            return module
        for sub_module in module.children():
            if isinstance(sub_module, torch.nn.Module):
                sub_module = cast_weight(sub_module)
        return module
    for _, module in model.named_modules():
        module = cast_weight(module)
    return model
