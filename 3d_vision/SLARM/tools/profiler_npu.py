# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu.profiler import profile, schedule
from torch_npu.profiler import ProfilerActivity, tensorboard_trace_handler


class NPUTorchProfileHook():
    def __init__(self, mode, skip_first=3, wait=3, warmup=3, active=1, repeat=1, with_stack=False, profiler_level="1"):
        my_schedule = schedule(skip_first=skip_first,
                               wait=wait,
                               warmup=warmup,
                               active=active,
                               repeat=repeat)
        profile_level_map = {
            "0": torch_npu.profiler.ProfilerLevel.Level0,
            "1": torch_npu.profiler.ProfilerLevel.Level1,
            "2": torch_npu.profiler.ProfilerLevel.Level2
        }

        # Resolve profiler level from parameter or environment variable
        level_str = os.environ.get('NPU_PROFILER_LEVEL', profiler_level)
        if level_str not in profile_level_map:
            raise ValueError(f"Invalid profiler_level '{level_str}'. Must be '0', '1', or '2'.")
        resolved_profiler_level = profile_level_map[level_str]

        profile_tb_logger = os.path.join("profile_results/{}".format(mode))
        os.makedirs(profile_tb_logger, exist_ok=True)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=resolved_profiler_level,
            l2_cache=(level_str == "2"),  # Only enable L2 cache at Level2
            data_simplification=False
        )

        self.profiler = profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.NPU,
            ],
            schedule=my_schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
            with_flops=False,
            experimental_config=experimental_config,
            # Expert parameter default level Level0, can set different Level as needed
            on_trace_ready=tensorboard_trace_handler(profile_tb_logger),
        )

    def before_train(self):
        self.profiler.__enter__()

    def after_step(self):
        self.profiler.step()

    def after_train(self):
        self.profiler.__exit__(None, None, None)
