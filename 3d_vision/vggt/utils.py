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

"""
Configuration Utilities

This module provides common utilities for loading YAML configuration files
and other helper functions for model inference scripts.
"""

import os
import logging
from dataclasses import dataclass

import yaml
import torch
import torch_npu

from vggt.models.vggt import VGGT
from quant.vggt_utils import replace_linear_in_vggt, set_ignore_quantize
from vggt.utils.cast_weight import cast_model_weight


@dataclass
class StandardModelConfig:
    """Configuration for loading standard VGGT model."""
    checkpoint_path: str
    device: str
    dtype: torch.dtype
    use_rope_cache: bool
    use_dpt_pos_embed_cache: bool
    optimization: dict


@dataclass
class ParallelConfigResult:
    """Result of parallel computation configuration setup."""
    enable_sp: bool
    ulysses_degree: int
    ring_degree: int
    sp_config: object
    ulysses_group: object
    ring_group: object
    global_group: object


@dataclass
class ModelLoadConfig:
    """Configuration for loading VGGT model with sequence parallel."""
    checkpoint_path: str
    sp_config: object
    ulysses_group: object
    ring_group: object
    global_group: object
    device: str
    dtype: torch.dtype
    use_rope_cache: bool
    use_dpt_pos_embed_cache: bool
    optimization: dict
    rank: int


@dataclass
class InferenceConfig:
    """Configuration for running inference loop."""
    model: object
    images: torch.Tensor
    profiler: object
    num_runs: int
    rank: int
    use_autocast: bool
    dtype: torch.dtype


@dataclass
class VGGTConfig:
    """VGGT inference configuration loaded from YAML."""
    # Basic arguments
    ckpt: str
    images_path: str = "examples/kitchen/images"
    
    # Profiling arguments
    enable_profiling: bool = False
    profile_dir: str = "prof_sp"
    
    # Performance arguments
    num_runs: int = 6
    
    # Optimization configuration (contains all optimization settings)
    optimization: dict = None


@dataclass
class W8A8ModelConfig:
    """Configuration for loading W8A8 quantized model."""
    checkpoint_path: str
    device: str
    use_rope_cache: bool
    use_dpt_pos_embed_cache: bool


def build_vggt_config(model_args: dict, optimization: dict) -> VGGTConfig:
    """
    Build VGGTConfig from model_args and optimization dictionaries.
    
    Args:
        model_args: Dictionary from load_yaml_config
        optimization: Dictionary from load_optimization_config (always valid)
    
    Returns:
        VGGTConfig object
    """
    config_dict = {}
    
    # Required argument
    config_dict['ckpt'] = model_args['ckpt']
    
    # Optional arguments with defaults
    config_dict['images_path'] = get_config_value(model_args, 'images-path', 'examples/kitchen/images')
    config_dict['enable_profiling'] = get_config_value(model_args, 'enable-profiling', False)
    config_dict['profile_dir'] = get_config_value(model_args, 'profile-dir', 'prof_sp')
    config_dict['num_runs'] = get_config_value(model_args, 'num-runs', 6)
    
    # Only store optimization config, no redundant fields
    config_dict['optimization'] = optimization
    
    return VGGTConfig(**config_dict)


def load_yaml_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    This function loads the entire YAML config and returns the full dictionary.
    Each script can then extract model_args and optimization from this dictionary.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing full YAML config (including model_args and optimization)
    
    Raises:
        FileNotFoundError: If YAML file does not exist
        ValueError: If required parameter 'ckpt' is not found
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    logging.info(f"[INFO] Loaded YAML config from: {config_path}")
    
    # Extract model_args from YAML
    model_args = config.get('model_args', {})
    
    # Required argument: ckpt
    if 'ckpt' not in model_args:
        raise ValueError("Required parameter 'ckpt' not found in YAML config")
    
    # Log loaded configuration
    logging.info("[INFO] Configuration loaded:")
    logging.info("[INFO]   model_args:")
    for key, value in model_args.items():
        logging.info(f"[INFO]     {key}: {value}")
    
    # Log optimization configuration if present
    if 'optimization' in config:
        logging.info("[INFO]   optimization:")
        for category, settings in config['optimization'].items():
            logging.info(f"[INFO]     {category}:")
            for key, value in settings.items():
                logging.info(f"[INFO]       {key}: {value}")
    
    return config


def get_config_value(model_args: dict, key: str, default=None, required: bool = False):
    """
    Get a configuration value from model_args dictionary.
    
    Handles kebab-case to snake_case conversion automatically.
    
    Args:
        model_args: Dictionary from load_yaml_config
        key: Configuration key (can be kebab-case or snake_case)
        default: Default value if key not found
        required: If True, raises ValueError when key not found
    
    Returns:
        Configuration value
    
    Raises:
        ValueError: If required=True and key not found
    """
    # Try kebab-case first (YAML convention), then snake_case
    value = model_args.get(key)
    if value is None:
        snake_key = key.replace('-', '_')
        value = model_args.get(snake_key)
    
    if value is None and required:
        raise ValueError(f"Required parameter '{key}' not found in YAML config")
    
    return value if value is not None else default


def load_optimization_config(config: dict, world_size: int = 1) -> dict:
    """
    Load optimization configuration from already-loaded YAML config dictionary.
    
    This function extracts the 'optimization' block from the config dictionary
    and performs validation to ensure the configuration is valid.
    
    Args:
        config: Dictionary from load_yaml_config (full YAML config)
        world_size: Number of cards (from YAML config or runtime)
    
    Returns:
        Dictionary containing optimization settings
    
    Raises:
        ValueError: If optimization configuration is invalid
    """
    # Extract optimization block from config
    optimization = config.get('optimization', {})
    
    if not optimization:
        logging.warning("[WARN] No 'optimization' block found in YAML, using default settings")
        optimization = get_default_optimization_config()
    
    # Validate optimization configuration
    validate_optimization_config(optimization, world_size)
    
    return optimization


def get_model_dtype(optimization: dict) -> torch.dtype:
    """
    Get model dtype from optimization config.
    
    Args:
        optimization: Optimization configuration dictionary (from YAML)
    
    Returns:
        torch.dtype: The model dtype (torch.float32 or torch.bfloat16)
    
    Raises:
        ValueError: If dtype is not in supported list
    """
    quantization = optimization.get('quantization', {})
    dtype_str = quantization.get('dtype', 'bf16')
    dtype_map = {
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported quantization.dtype: {dtype_str}, "
            f"must be one of {list(dtype_map.keys())}"
        )
    dtype = dtype_map[dtype_str]
    logging.info(f"[OPTIMIZATION] quantization.dtype: {dtype_str} ({dtype})")
    return dtype


def get_default_optimization_config() -> dict:
    """
    Get default optimization configuration.
    
    Returns:
        Dictionary with default optimization settings (all optimizations enabled)
    """
    return {
        'computation-redundancy-elimination': {
            'rope-cache': True,
            'dpt-pos-embed-cache': True,
            'cos-sin-dtype-optimization': True,
        },
        'parallel-computation': {
            'enable': False,
            'ulysses-degree': 1,
            'ring-degree': 1,
        },
        'quantization': {
            'dtype': 'bf16',
            'int8-w8a8': {
                'enable': False,
                'build': False,
            },
        },
        'memory-and-data-format': {
            'conv-weight-layout-preconvert': True,
        },
    }


def _validate_redundancy_elimination(redundancy_elim: dict) -> None:
    """Validate computation-redundancy-elimination configuration block."""
    for key in ['rope-cache', 'dpt-pos-embed-cache', 'cos-sin-dtype-optimization']:
        if key in redundancy_elim:
            if not isinstance(redundancy_elim[key], bool):
                raise ValueError(f"optimization.computation-redundancy-elimination.{key} must be a boolean")


def _validate_parallel_computation(parallel: dict) -> None:
    """Validate parallel-computation configuration block."""
    if 'enable' in parallel:
        if not isinstance(parallel['enable'], bool):
            raise ValueError("optimization.parallel-computation.enable must be a boolean")

    if 'ulysses-degree' in parallel:
        if not isinstance(parallel['ulysses-degree'], int) or parallel['ulysses-degree'] < 1:
            raise ValueError("optimization.parallel-computation.ulysses-degree must be a positive integer")

    if 'ring-degree' in parallel:
        if not isinstance(parallel['ring-degree'], int) or parallel['ring-degree'] < 1:
            raise ValueError("optimization.parallel-computation.ring-degree must be a positive integer")


def _validate_parallel_constraints(parallel: dict, world_size: int) -> None:
    """Validate parallel computation constraints."""
    if parallel.get('enable', False):
        ulysses_degree = parallel.get('ulysses-degree', 1)
        ring_degree = parallel.get('ring-degree', 1)
        if ulysses_degree * ring_degree != world_size:
            raise ValueError(
                f"Parallel computation constraint violated: "
                f"ulysses-degree ({ulysses_degree}) × ring-degree ({ring_degree}) "
                f"must equal world_size ({world_size})"
            )

    if world_size == 1 and parallel.get('enable', False):
        raise ValueError(
            "parallel-computation.enable must be False for single-card inference (world_size=1)"
        )


def _validate_quantization(quantization: dict) -> None:
    """Validate quantization configuration block."""
    if 'dtype' in quantization:
        valid_dtypes = ['fp32', 'bf16']
        if quantization['dtype'] not in valid_dtypes:
            raise ValueError(f"optimization.quantization.dtype must be one of {valid_dtypes}")

    int8_config = quantization.get('int8-w8a8', {})
    if 'enable' in int8_config:
        if not isinstance(int8_config['enable'], bool):
            raise ValueError("optimization.quantization.int8-w8a8.enable must be a boolean")
    if 'build' in int8_config:
        if not isinstance(int8_config['build'], bool):
            raise ValueError("optimization.quantization.int8-w8a8.build must be a boolean")


def _validate_memory_format(memory_format: dict) -> None:
    """Validate memory-and-data-format configuration block."""
    if 'conv-weight-layout-preconvert' in memory_format:
        if not isinstance(memory_format['conv-weight-layout-preconvert'], bool):
            raise ValueError("optimization.memory-and-data-format.conv-weight-layout-preconvert must be a boolean")


def validate_optimization_config(optimization: dict, world_size: int) -> None:
    """
    Validate optimization configuration.

    Checks for:
    - Required fields existence
    - Parameter type correctness
    - Parameter value validity
    - Parameter combination constraints

    Args:
        optimization: Optimization configuration dictionary
        world_size: Number of cards (from YAML config)

    Raises:
        ValueError: If configuration is invalid
    """
    _validate_redundancy_elimination(optimization.get('computation-redundancy-elimination', {}))
    _validate_parallel_computation(optimization.get('parallel-computation', {}))
    _validate_parallel_constraints(optimization.get('parallel-computation', {}), world_size)
    _validate_quantization(optimization.get('quantization', {}))
    _validate_memory_format(optimization.get('memory-and-data-format', {}))


# -------------------------------------------------------------------------
# Model Loading Utilities
# -------------------------------------------------------------------------

def load_w8a8_model(config: W8A8ModelConfig):
    """
    Load W8A8 quantized model.

    Args:
        config: W8A8ModelConfig object containing all model loading parameters

    Returns:
        Loaded W8A8 model
    """
    logging.info("Loading W8A8 quantized model...")
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.to(config.device).eval()
        logging.info("W8A8 quantized model object loaded successfully")
    elif isinstance(checkpoint, dict):
        logging.info("Checkpoint is a state_dict; building W8A8 model from VGGT weights...")
        model = VGGT(
            use_rope_cache=config.use_rope_cache,
            use_dpt_pos_embed_cache=config.use_dpt_pos_embed_cache,
        )
        model.load_state_dict(checkpoint)
        model = model.bfloat16()
        model.to(config.device).eval()
        set_ignore_quantize(model, ignore_quantize=True)
        model = replace_linear_in_vggt(model, device=config.device)
        logging.info("W8A8 model built from state_dict successfully")
    else:
        raise TypeError(f"Unsupported W8A8 checkpoint type: {type(checkpoint)}")

    return model


def load_standard_model(config: StandardModelConfig):
    """
    Load standard model.

    Args:
        config: StandardModelConfig object containing all model loading parameters

    Returns:
        Loaded standard model
    """
    logging.info("Loading standard model...")
    model = VGGT(
        use_rope_cache=config.use_rope_cache,
        use_dpt_pos_embed_cache=config.use_dpt_pos_embed_cache,
    )
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint)

    if config.dtype == torch.float32:
        model = model.float()
    else:
        model = model.bfloat16()

    model.to(config.device).eval()

    memory_format = config.optimization.get('memory-and-data-format', {})
    # 950 only supports ND format, no need for NZ format conversion
    if memory_format.get('conv-weight-layout-preconvert', True) and not is_ascend_950():
        model = cast_model_weight(model)
        logging.info("[OPTIMIZATION] conv-weight-layout-preconvert: enabled")

    logging.info("Standard model loaded successfully")
    return model


def build_and_save_w8a8_model(model, device):
    """
    Build W8A8 quantized model and save to file.

    Args:
        model: Standard model to convert
        device: Device for conversion

    Returns:
        None (function saves model to file)
    """
    logging.info("Building W8A8 quantized model...")
    set_ignore_quantize(model, ignore_quantize=True)
    replace_linear_in_vggt(model, device=device)
    save_path = os.path.join(os.getcwd(), "VGGT_model_W8A8.pt")
    torch.save(model, save_path)
    logging.info(f"W8A8 model saved to {save_path}")


def is_ascend_950() -> bool:
    """Check whether the current NPU device is an Ascend 950 model."""
    return "950" in torch.npu.get_device_name()