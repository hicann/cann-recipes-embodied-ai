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

import argparse
from dataclasses import dataclass
import logging
import time

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from infer_utils import get_device, make_dummy_observation, move_to_device_and_dtype, synchronize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

try:
    import torch_npu  # noqa: F401

    logger.info("torch_npu imported successfully.")
except ImportError as import_err:
    logger.warning("torch_npu import failed, continue without NPU support: %s", import_err)


@dataclass
class RuntimeEnv:
    device: torch.device
    dtype: torch.dtype


@dataclass
class InferencePipeline:
    policy: PI05Policy
    preprocess: object
    postprocess: object


@dataclass
class InferenceStats:
    model_path: str
    num_inference: int
    total_cpu_wall_time: float
    action: object


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end inference example for PI0.5 Policy")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the pretrained model or Hugging Face model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu",
        help="Device to run inference on (e.g. npu, npu:1, cuda, cuda:1, cpu, mps)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for inference",
    )
    parser.add_argument("--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument(
        "--num_inference",
        type=int,
        default=3,
        help="Number of inference iterations for timing",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Invalid dtype: {dtype_name}")
    return dtype


def load_policy(model_path, device, dtype):
    logger.info("Loading model from %s...", model_path)
    policy = PI05Policy.from_pretrained(model_path)
    policy.to(device)
    policy.to(dtype)
    policy.eval()
    logger.info("Model loaded successfully.")
    return policy


def run_single_inference(pipeline, runtime_env, reset_policy=False):
    obs = make_dummy_observation(batch_size=1, task="Pick up the object\n")
    obs = pipeline.preprocess(obs)
    obs = move_to_device_and_dtype(obs, runtime_env.device, runtime_env.dtype)

    if reset_policy:
        pipeline.policy.reset()

    with torch.no_grad():
        action = pipeline.policy.select_action(obs)

    synchronize(runtime_env.device, logger)
    return pipeline.postprocess(action)


def run_warmup(pipeline, runtime_env, num_warmup):
    logger.info("Starting warm-up (%s iterations)...", num_warmup)
    action = None
    for _ in range(num_warmup):
        action = run_single_inference(pipeline, runtime_env, reset_policy=False)
    logger.info("Warm-up completed.")
    return action


def run_timed_inference(pipeline, runtime_env, num_inference):
    logger.info("Starting inference timing (%s iterations)...", num_inference)
    total_cpu_wall_time = 0.0
    action = None

    for _ in range(num_inference):
        start_time = time.perf_counter()
        action = run_single_inference(pipeline, runtime_env, reset_policy=True)
        end_time = time.perf_counter()
        total_cpu_wall_time += end_time - start_time

    return total_cpu_wall_time, action


def log_inference_results(runtime_env, stats):
    avg_time_ms = (stats.total_cpu_wall_time / stats.num_inference) * 1000
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    logger.info("----------------------------------------")
    logger.info("Inference Results for %s", stats.model_path)
    logger.info("Device: %s, Dtype: %s", runtime_env.device, runtime_env.dtype)

    if isinstance(stats.action, torch.Tensor):
        logger.info("Action shape: %s", stats.action.shape)
    elif isinstance(stats.action, dict) and "action" in stats.action:
        logger.info("Action shape: %s", stats.action["action"].shape)
    else:
        logger.info("Action type: %s", type(stats.action))

    logger.info(
        "Total time for %s runs: %.4f s",
        stats.num_inference,
        stats.total_cpu_wall_time,
    )
    logger.info("Average latency: %.4f ms", avg_time_ms)
    logger.info("Throughput: %.2f FPS", fps)
    logger.info("----------------------------------------")


def main():
    args = parse_args()

    try:
        device = get_device(args.device, logger)
        dtype = resolve_dtype(args.dtype)
    except ValueError as exc:
        logger.error(str(exc))
        raise SystemExit(2) from exc

    logger.info("Using device: %s", device)
    logger.info("Using dtype: %s", dtype)

    try:
        policy = load_policy(args.pretrained_model_name_or_path, device, dtype)
    except Exception as load_err:
        logger.exception("Failed to load model: %s", load_err)
        raise SystemExit(1) from load_err

    preprocess, postprocess = make_pre_post_processors(policy.config)
    runtime_env = RuntimeEnv(device=device, dtype=dtype)
    pipeline = InferencePipeline(policy=policy, preprocess=preprocess, postprocess=postprocess)

    run_warmup(pipeline, runtime_env, args.num_warmup)
    total_cpu_wall_time, action = run_timed_inference(pipeline, runtime_env, args.num_inference)
    stats = InferenceStats(
        model_path=args.pretrained_model_name_or_path,
        num_inference=args.num_inference,
        total_cpu_wall_time=total_cpu_wall_time,
        action=action,
    )
    log_inference_results(runtime_env, stats)


if __name__ == "__main__":
    main()
