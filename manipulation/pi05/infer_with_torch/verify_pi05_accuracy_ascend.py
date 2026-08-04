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
import logging
import time

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from infer_utils import get_device, make_dummy_observation, move_to_device, synchronize

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
except ImportError as exc:
    logger.warning("torch_npu import failed, NPU verification may be skipped: %s", exc)


def run_inference(policy, obs, device, noise=None, dtype=torch.float32):
    logger.info("Moving policy to %s with dtype %s...", device, dtype)
    policy.to(device)
    policy.to(dtype)
    policy.eval()

    obs_device = move_to_device(obs, device)

    noise_device = None
    if noise is not None:
        noise_device = noise.to(device).to(dtype)

    policy.reset()
    with torch.no_grad():
        action = policy.predict_action_chunk(obs_device, noise=noise_device)
    synchronize(device, logger)

    return action.cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=1
    ).item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify PI0.5 inference consistency between CPU and NPU"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or Hugging Face model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu:0",
        help="NPU device to run inference on (e.g. npu, npu:1)",
    )
    return parser.parse_args()


def load_policy(model_path):
    logger.info("Loading model from %s...", model_path)
    policy = PI05Policy.from_pretrained(model_path)
    policy.eval()
    logger.info("Model loaded successfully.")
    return policy


def prepare_observation(policy):
    preprocess, _postprocess = make_pre_post_processors(policy.config)
    batch_size = 1
    image_shape = (3, 224, 224)
    raw_obs = make_dummy_observation(
        batch_size=batch_size,
        image_shape=image_shape,
        task=["Pick up the object"] * batch_size,
    )
    obs = preprocess(raw_obs)
    img_key = "observation.images.base_0_rgb"
    if img_key in obs:
        logger.info(
            "Preprocessed image dtype: %s, min: %s, max: %s",
            obs[img_key].dtype,
            obs[img_key].min(),
            obs[img_key].max(),
        )
    return obs, batch_size


def compare_actions(cpu_action, npu_action, chunk_size):
    cpu_vec = cpu_action.float().contiguous()
    npu_vec = npu_action.float().contiguous()
    if not torch.isfinite(cpu_vec).all():
        logger.error("CPU output contains NaN/Inf, similarity is invalid.")
        return False
    if not torch.isfinite(npu_vec).all():
        logger.error("NPU output contains NaN/Inf, similarity is invalid.")
        return False

    cpu_flat = cpu_vec.flatten()
    npu_flat = npu_vec.flatten()
    global_cosine_sim = cosine_sim(cpu_flat, npu_flat)
    logger.info("Global Cosine Similarity: %.6f", global_cosine_sim)

    logger.info("Per-timestep Cosine Similarity:")
    step_similarities = torch.nn.functional.cosine_similarity(cpu_vec, npu_vec, dim=-1)
    mean_step_similarities = step_similarities.mean(dim=0)  # [chunk_size]
    min_sim = mean_step_similarities.min().item()
    avg_sim = mean_step_similarities.mean().item()

    for t in range(min(5, chunk_size)):
        logger.info("  Step %s: %.6f", t, mean_step_similarities[t].item())
    if chunk_size > 10:
        logger.info("  ...")
        for t in range(chunk_size - 5, chunk_size):
            logger.info("  Step %s: %.6f", t, mean_step_similarities[t].item())

    logger.info("  Minimum Per-step Similarity: %.6f", min_sim)
    logger.info("  Average Per-step Similarity: %.6f", avg_sim)

    mse = torch.nn.functional.mse_loss(cpu_vec, npu_vec)
    logger.info("MSE Loss: %.6f", mse.item())

    if global_cosine_sim > 0.99 and min_sim > 0.99:
        logger.info("Verification SUCCESS: All similarities > 0.99")
        return True

    logger.error("Verification FAILED: Similarity check failed")
    return False


def main():
    args = parse_args()

    try:
        policy = load_policy(args.pretrained_model_name_or_path)
    except Exception as load_err:
        logger.error("Failed to load model: %s", load_err)
        return

    obs, batch_size = prepare_observation(policy)
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim
    noise_shape = (batch_size, chunk_size, max_action_dim)
    logger.info("Generating fixed noise with shape %s...", noise_shape)

    fixed_noise = torch.randn(noise_shape, dtype=torch.float32)

    cpu_dtype = torch.float32
    npu_dtype = torch.float16

    logger.info("Running inference on CPU...")
    start_time = time.time()
    cpu_action = run_inference(
        policy, obs, torch.device("cpu"), noise=fixed_noise, dtype=cpu_dtype
    )
    logger.info("CPU inference done in %.4fs", time.time() - start_time)
    logger.info("CPU action shape: %s", cpu_action.shape)

    device_npu = get_device(args.device, logger)
    if device_npu.type == "npu":
        logger.info("Running inference on NPU...")
        try:
            start_time = time.time()
            npu_action = run_inference(
                policy, obs, device_npu, noise=fixed_noise, dtype=npu_dtype
            )
            logger.info("NPU inference done in %.4fs", time.time() - start_time)
            logger.info("NPU action shape: %s", npu_action.shape)
            compare_actions(cpu_action, npu_action, chunk_size)
        except Exception as npu_err:
            logger.exception("NPU inference failed: %s", npu_err)
    else:
        logger.warning("NPU not available. Skipping NPU run for device '%s'.", args.device)


if __name__ == "__main__":
    main()
