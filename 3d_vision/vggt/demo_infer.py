# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

import os
import argparse
import contextlib
import time
import logging

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu

from vggt.models.vggt import VGGT
from vggt.sp import SPConfig
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.cast_weight import cast_model_weight
from vggt.heads.utils import set_cos_sin_dtype_optimization_enabled
from eval.general_utils import fix_random_seed
from quant.vggt_utils import replace_linear_in_vggt, set_ignore_quantize
from quant.vggt_linear import LinearW8A8
from utils import (
    load_yaml_config,
    load_optimization_config,
    get_model_dtype,
    load_w8a8_model,
    load_standard_model,
    build_and_save_w8a8_model,
    StandardModelConfig,
    W8A8ModelConfig,
    ParallelConfigResult,
    ModelLoadConfig,
    InferenceConfig,
    build_vggt_config,
    is_ascend_950
)

logging.basicConfig(level=logging.INFO)


class EmptyContextManager(contextlib.nullcontext):
    """Empty context manager that is used when profiling is disabled."""

    @staticmethod
    def step():
        pass


def define_profiler(enable_profiler=False, profile_save_path="prof", rank=0):
    """
    Define profiler based on configuration.
    
    Args:
        enable_profiler: Whether to enable profiling
        profile_save_path: Directory to save profiling results
        rank: Rank ID for multi-card scenario
    
    Returns:
        Profiler context manager
    """
    if enable_profiler:
        os.makedirs(profile_save_path, exist_ok=True)
        
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False
        )
        
        profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU,
                torch_npu.profiler.ProfilerActivity.CPU,
            ],
            with_stack=False,
            record_shapes=True,
            profile_memory=True,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=2,
                active=1,
                repeat=1,
                skip_first=0
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path),
            with_modules=False,
            with_flops=False,
        )
        logging.info(f"Profiler enabled, results will be saved to {profile_save_path}")
    else:
        profiler = EmptyContextManager()
    
    return profiler


def setup_distributed():
    """Initialize distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        try:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        except ValueError as exc:
            raise RuntimeError("Invalid distributed environment variables.") from exc
    else:
        raise RuntimeError("Distributed environment not set up properly")
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='hccl')
    
    logging.info(f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank


def setup_sequence_parallel_groups(ulysses_degree, ring_degree):
    """
    Create process groups for sequence parallel.
    
    Args:
        ulysses_degree: Degree of Ulysses parallelism
        ring_degree: Degree of Ring Attention parallelism
    
    Returns:
        sp_config, ulysses_group, ring_group, global_group
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if world_size != ulysses_degree * ring_degree:
        raise ValueError(
            f"Runtime world_size ({world_size}) does not match YAML config "
            f"(ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, "
            f"expected world_size={ulysses_degree * ring_degree}). "
            f"Please check torchrun --nproc_per_node matches YAML world_size."
        )
    
    # Create Ulysses groups
    ulysses_groups = []
    for i in range(ring_degree):
        ranks = [i * ulysses_degree + j for j in range(ulysses_degree)]
        group = dist.new_group(ranks)
        ulysses_groups.append(group)
    
    ulysses_rank = rank % ulysses_degree
    ulysses_group_idx = rank // ulysses_degree
    ulysses_group = ulysses_groups[ulysses_group_idx]
    
    # Create Ring groups
    ring_groups = []
    for i in range(ulysses_degree):
        ranks = [i + j * ulysses_degree for j in range(ring_degree)]
        group = dist.new_group(ranks)
        ring_groups.append(group)
    
    ring_group = ring_groups[ulysses_rank]
    
    global_group = dist.group.WORLD
    
    sp_config = SPConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        use_ring_overlap=True,
    )
    
    logging.info(f"Sequence Parallel setup: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}")
    
    return sp_config, ulysses_group, ring_group, global_group


def get_all_files_paths(dir_path):
    """Get all file paths in a directory."""
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def sync_and_get_time(start_time=None, use_syn=True, log_result=False):
    """Synchronize NPU and get timestamp."""
    if use_syn:
        torch.npu.synchronize()
    timestamp = time.time()
    if start_time is not None:
        timestamp -= start_time
        if log_result:
            logging.info(f"VGGT inference time cost is: {timestamp*1000:.2f} ms")
        return timestamp
    return timestamp


def _setup_parallel_config(optimization: dict) -> ParallelConfigResult:
    """
    Setup parallel computation configuration.

    Args:
        optimization: Optimization configuration dictionary

    Returns:
        ParallelConfigResult object containing all parallel configuration values
    """
    parallel = optimization.get('parallel-computation', {})
    enable_sp = parallel.get('enable', False)
    ulysses_degree = parallel.get('ulysses-degree', 2)
    ring_degree = parallel.get('ring-degree', 2)

    if enable_sp:
        sp_config, ulysses_group, ring_group, global_group = setup_sequence_parallel_groups(
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree
        )
    else:
        sp_config = None
        ulysses_group = None
        ring_group = None
        global_group = None

    return ParallelConfigResult(
        enable_sp=enable_sp,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        sp_config=sp_config,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        global_group=global_group
    )


def _setup_redundancy_elimination(optimization: dict, rank: int):
    """
    Setup computation redundancy elimination configuration.

    Args:
        optimization: Optimization configuration dictionary
        rank: Current process rank

    Returns:
        Tuple of (use_rope_cache, use_dpt_pos_embed_cache, use_cos_sin_dtype_opt)
    """
    redundancy_elim = optimization.get('computation-redundancy-elimination', {})
    use_rope_cache = redundancy_elim.get('rope-cache', True)
    use_dpt_pos_embed_cache = redundancy_elim.get('dpt-pos-embed-cache', True)
    use_cos_sin_dtype_opt = redundancy_elim.get('cos-sin-dtype-optimization', True)

    set_cos_sin_dtype_optimization_enabled(use_cos_sin_dtype_opt)

    if rank == 0:
        logging.info(f"[OPTIMIZATION] rope-cache: {use_rope_cache}")
        logging.info(f"[OPTIMIZATION] dpt-pos-embed-cache: {use_dpt_pos_embed_cache}")
        logging.info(f"[OPTIMIZATION] cos-sin-dtype-optimization: {use_cos_sin_dtype_opt}")

    return use_rope_cache, use_dpt_pos_embed_cache, use_cos_sin_dtype_opt


def _load_model_with_sp(config: ModelLoadConfig):
    """
    Load VGGT model with sequence parallel support.

    Args:
        config: ModelLoadConfig object containing all model loading parameters

    Returns:
        Loaded VGGT model
    """
    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        sp_config=config.sp_config,
        sp_ulysses_group=config.ulysses_group,
        sp_ring_group=config.ring_group,
        sp_global_group=config.global_group,
        use_rope_cache=config.use_rope_cache,
        use_dpt_pos_embed_cache=config.use_dpt_pos_embed_cache,
    )

    checkpoint = torch.load(config.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    if config.dtype == torch.float32:
        model = model.float()
    else:
        model = model.bfloat16()

    model.to(config.device)
    model.eval()

    memory_format = config.optimization.get('memory-and-data-format', {})
    # 950 only supports ND format, no need for NZ format conversion
    if memory_format.get('conv-weight-layout-preconvert', True) and not is_ascend_950():
        model = cast_model_weight(model)
        if config.rank == 0:
            logging.info("[OPTIMIZATION] conv-weight-layout-preconvert: enabled")

    logging.info(f"Model loaded successfully on rank {config.rank}")
    return model


def _run_sp_inference_loop(config: InferenceConfig):
    """
    Run inference loop with sequence parallel.

    Args:
        config: InferenceConfig object containing all inference parameters

    Returns:
        List of execution times
    """
    exec_time_list = []

    # Combine context managers to reduce nesting depth
    inference_context = torch.cuda.amp.autocast(enabled=config.use_autocast, dtype=config.dtype)

    with config.profiler:
        with torch.no_grad(), inference_context:
            for step in range(config.num_runs):
                dist.barrier()
                start_time = sync_and_get_time()
                predictions = config.model(config.images)
                dist.barrier()
                exec_time = sync_and_get_time(start_time, log_result=(config.rank == 0 and step >= 2))
                exec_time_list.append(exec_time)
                config.profiler.step()

    return exec_time_list


def _load_images_for_inference(images_path, device, dtype):
    """
    Load and preprocess images for inference.

    Args:
        images_path: Path to images directory
        device: Target device
        dtype: Target data type

    Returns:
        Tuple of (images tensor, number of images)
    """
    image_names = sorted(get_all_files_paths(images_path))
    images = load_and_preprocess_images(image_names).to(device=device, dtype=dtype)
    if len(images.shape) == 4:
        images = images.unsqueeze(0)
    logging.info(f"Loaded {len(image_names)} images, shape: {images.shape}")
    return images, len(image_names)


def _warmup_model(model, images, use_autocast, dtype):
    """
    Perform model warmup before inference.

    Args:
        model: VGGT model
        images: Input images tensor
        use_autocast: Whether to use autocast
        dtype: Model data type
    """
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_autocast, dtype=dtype):
            _ = model(images)
    dist.barrier()
    logging.info("Warmup completed")


def _setup_profiler_for_sp(enable_profiling, profile_dir, rank):
    """
    Setup profiler for sequence parallel inference.

    Args:
        enable_profiling: Whether to enable profiling
        profile_dir: Base directory for profiling results
        rank: Current process rank

    Returns:
        Profiler context manager
    """
    profile_save_path = os.path.join(profile_dir, f"rank_{rank}") if enable_profiling else "prof"
    profiler = define_profiler(
        enable_profiler=enable_profiling,
        profile_save_path=profile_save_path,
        rank=rank
    )
    return profiler


def _log_inference_results(exec_time_list, rank):
    """
    Log inference timing results.

    Args:
        exec_time_list: List of execution times
        rank: Current process rank
    """
    if rank == 0:
        valid_times = exec_time_list[2:]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        logging.info(f"Execution times (ms): {[t*1000 for t in exec_time_list]}")
        logging.info(f"Average inference time (excluding warmup): {avg_time*1000:.2f} ms")


def run_inference_with_sp(args):
    """Main inference function with sequence parallel support."""
    fix_random_seed(42)

    rank, world_size, local_rank = setup_distributed()
    device = f"npu:{local_rank}"

    parallel_result = _setup_parallel_config(args.optimization)
    dtype = get_model_dtype(args.optimization)
    use_rope_cache, use_dpt_pos_embed_cache, _ = _setup_redundancy_elimination(args.optimization, rank)

    model_load_config = ModelLoadConfig(
        checkpoint_path=args.ckpt,
        sp_config=parallel_result.sp_config,
        ulysses_group=parallel_result.ulysses_group,
        ring_group=parallel_result.ring_group,
        global_group=parallel_result.global_group,
        device=device,
        dtype=dtype,
        use_rope_cache=use_rope_cache,
        use_dpt_pos_embed_cache=use_dpt_pos_embed_cache,
        optimization=args.optimization,
        rank=rank
    )
    model = _load_model_with_sp(model_load_config)

    images, _ = _load_images_for_inference(args.images_path, device, dtype)

    use_autocast = (dtype == torch.bfloat16)
    _warmup_model(model, images, use_autocast, dtype)

    profiler = _setup_profiler_for_sp(args.enable_profiling, args.profile_dir, rank)

    inference_config = InferenceConfig(
        model=model,
        images=images,
        profiler=profiler,
        num_runs=args.num_runs,
        rank=rank,
        use_autocast=use_autocast,
        dtype=dtype
    )
    exec_time_list = _run_sp_inference_loop(inference_config)

    dist.barrier()
    _log_inference_results(exec_time_list, rank)
    dist.destroy_process_group()


def _get_quantization_config(optimization: dict):
    """
    Get quantization configuration.

    Args:
        optimization: Optimization configuration dictionary

    Returns:
        Tuple of (enable_w8a8, build_w8a8)
    """
    quantization = optimization.get('quantization', {})
    int8_config = quantization.get('int8-w8a8', {})
    enable_w8a8 = int8_config.get('enable', False)
    build_w8a8 = int8_config.get('build', False)
    logging.info(f"[OPTIMIZATION] quantization.int8-w8a8.enable: {enable_w8a8}")
    logging.info(f"[OPTIMIZATION] quantization.int8-w8a8.build: {build_w8a8}")
    return enable_w8a8, build_w8a8


def _setup_redundancy_elimination_single(optimization: dict):
    """
    Setup computation redundancy elimination configuration for single card.

    Args:
        optimization: Optimization configuration dictionary

    Returns:
        Tuple of (use_rope_cache, use_dpt_pos_embed_cache)
    """
    redundancy_elim = optimization.get('computation-redundancy-elimination', {})
    use_rope_cache = redundancy_elim.get('rope-cache', True)
    use_dpt_pos_embed_cache = redundancy_elim.get('dpt-pos-embed-cache', True)
    use_cos_sin_dtype_opt = redundancy_elim.get('cos-sin-dtype-optimization', True)

    set_cos_sin_dtype_optimization_enabled(use_cos_sin_dtype_opt)
    logging.info(f"[OPTIMIZATION] rope-cache: {use_rope_cache}")
    logging.info(f"[OPTIMIZATION] dpt-pos-embed-cache: {use_dpt_pos_embed_cache}")
    logging.info(f"[OPTIMIZATION] cos-sin-dtype-optimization: {use_cos_sin_dtype_opt}")

    return use_rope_cache, use_dpt_pos_embed_cache


def _run_single_inference(model, images, num_runs, use_autocast, dtype):
    """
    Run single card inference loop.

    Args:
        model: VGGT model
        images: Input images tensor
        num_runs: Number of inference runs
        use_autocast: Whether to use autocast
        dtype: Model data type

    Returns:
        List of execution times
    """
    exec_time_list = []

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_autocast, dtype=dtype):
            # Warmup
            predictions = model(images)

            for _ in range(num_runs):
                start_time = sync_and_get_time()
                predictions = model(images)
                exec_time = sync_and_get_time(start_time, log_result=True)
                exec_time_list.append(exec_time)

    return exec_time_list


def quick_start(args):
    """Single card inference with optional quantization."""
    fix_random_seed(42)

    # Device check
    device = "npu:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("NPU is not available. Check your environment.")
    logging.info(f"Using device: {device}")

    # Setup configurations
    enable_w8a8, build_w8a8 = _get_quantization_config(args.optimization)
    dtype = get_model_dtype(args.optimization)
    use_rope_cache, use_dpt_pos_embed_cache = _setup_redundancy_elimination_single(args.optimization)

    # Load model
    if enable_w8a8:
        w8a8_config = W8A8ModelConfig(
            checkpoint_path=args.ckpt,
            device=device,
            use_rope_cache=use_rope_cache,
            use_dpt_pos_embed_cache=use_dpt_pos_embed_cache
        )
        model = load_w8a8_model(w8a8_config)
    else:
        model_config = StandardModelConfig(
            checkpoint_path=args.ckpt,
            device=device,
            dtype=dtype,
            use_rope_cache=use_rope_cache,
            use_dpt_pos_embed_cache=use_dpt_pos_embed_cache,
            optimization=args.optimization
        )
        model = load_standard_model(model_config)
        if build_w8a8:
            build_and_save_w8a8_model(model, device)
            return

    # Load images
    image_names = sorted(get_all_files_paths(args.images_path))
    logging.info(f"Loading {len(image_names)} images from {args.images_path}")
    images = load_and_preprocess_images(image_names).to(device=device, dtype=dtype)

    # Run inference
    use_autocast = (dtype == torch.bfloat16)
    exec_time_list = _run_single_inference(model, images, args.num_runs, use_autocast, dtype)

    avg_time = sum(exec_time_list) / len(exec_time_list)
    logging.info(f"Execution times (ms): {[t*1000 for t in exec_time_list]}")
    logging.info(f"Average inference time: {avg_time*1000:.2f} ms ({avg_time:.4f} s)")


def parse_args():
    """Parse YAML configuration file path with default."""
    parser = argparse.ArgumentParser(
        "VGGT Inference (YAML Config Mode)",
        description="VGGT inference using YAML configuration file",
        add_help=True
    )
    
    parser.add_argument(
        "--config",
        default="config/single.yaml",
        help="YAML configuration file path (default: config/single.yaml)"
    )
    
    args = parser.parse_args()
    
    logging.info(f"[INFO] Using config file: {args.config}")
    
    # Load full YAML config (single read)
    config = load_yaml_config(args.config)
    
    # Extract model_args from config
    model_args = config.get('model_args', {})
    
    # Extract world_size from config
    world_size = config.get('world_size', 1)
    
    # Load optimization config from the same config dictionary
    optimization = load_optimization_config(config, world_size)
    
    # Build VGGTConfig from model_args and optimization
    return build_vggt_config(model_args, optimization)


def main():
    args = parse_args()
    
    # Check if running in distributed mode
    parallel = args.optimization.get('parallel-computation', {})
    enable_sp = parallel.get('enable', False)
    
    if enable_sp or ('RANK' in os.environ and 'WORLD_SIZE' in os.environ):
        logging.info("Running in distributed mode with sequence parallel")
        run_inference_with_sp(args)
    else:
        logging.info("Running in single card mode")
        quick_start(args)


if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=False)
    main()
    logging.info("Run all examples success")