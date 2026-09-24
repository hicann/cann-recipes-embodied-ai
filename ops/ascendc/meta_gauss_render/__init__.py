# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

__all__ = [
    "projection_three_dims_gaussian_fused",
    "calc_render",
    "gaussian_sort",
    "get_render_schedule",
    "spherical_harmonics",
    "flash_gaussian_build_mask",
    "gaussian_filter",
]

import os
import meta_gauss_render._C

from .ops.projection_three_dims_gaussian_fused import projection_three_dims_gaussian_fused
from .ops.calc_render import calc_render
from .ops.gaussian_sort import gaussian_sort
from .ops.get_render_schedule import get_render_schedule
from .ops.spherical_harmonics import spherical_harmonics
from .ops.flash_gaussian_build_mask import flash_gaussian_build_mask
from .ops.gaussian_filter import gaussian_filter


def _set_env():
    meta_gauss_render_root = os.path.dirname(os.path.abspath(__file__))
    meta_gauss_render_opp_path = os.path.join(meta_gauss_render_root, "packages", "vendors", "customize")
    ascend_custom_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    ascend_custom_opp_path = (
        meta_gauss_render_opp_path
        if not ascend_custom_opp_path
        else meta_gauss_render_opp_path + ":" + ascend_custom_opp_path
    )
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ascend_custom_opp_path
    meta_gauss_render_op_api_so_path = os.path.join(meta_gauss_render_opp_path, "op_api", "lib", "libcust_opapi.so")
    meta_gauss_render._C._init_op_api_so_path(meta_gauss_render_op_api_so_path)


_set_env()