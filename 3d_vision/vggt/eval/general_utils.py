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

import argparse
import random
import time
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class SPConfig:
    """Sequence Parallel Configuration"""
    ulysses_degree: int = 1
    ring_degree: int = 1
    use_ring_overlap: bool = True

    @property
    def sp_degree(self):
        return self.ulysses_degree * self.ring_degree


def setup_distributed(args):
    """Initialize distributed environment"""
    if not dist.is_initialized():
        dist.init_process_group(backend='hccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.ulysses_degree * args.ring_degree != world_size:
        raise ValueError(
            f"ulysses_degree ({args.ulysses_degree}) * ring_degree ({args.ring_degree}) "
            f"must equal world_size ({world_size})"
        )

    ulysses_pg = None
    ring_pg = None
    global_pg = None

    global_pg = dist.new_group(ranks=list(range(world_size)))

    if args.ulysses_degree > 1:
        for i in range(args.ring_degree):
            start_rank = i * args.ulysses_degree
            ranks = list(range(start_rank, start_rank + args.ulysses_degree))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                ulysses_pg = group

    if args.ring_degree > 1:
        for i in range(args.ulysses_degree):
            ranks = list(range(i, world_size, args.ulysses_degree))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                ring_pg = group

    return rank, world_size, ulysses_pg, ring_pg, global_pg


def fix_random_seed(seed=42):
    """Fix random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_depth_estimation_opts():
    """Command-line arguments for depth estimation."""
    parser = argparse.ArgumentParser(
        description="Test VGGT depth estimation on DTU datasets.", 
        add_help=True
    )
    
    parser.add_argument('--n_views', type=int, default=5, help='num of view')
    parser.add_argument('--levels', type=int, default=4, help='num of stages')
    parser.add_argument('--depth_conf_thres', type=int, default=3, 
        help='the range of depth conf is [1, +inf]. Usually > 3 or > 5 should be good enough')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')
    parser.add_argument('--testpath', help='data path for testing')
    parser.add_argument('--testlist', help='data list for testing')
    parser.add_argument("--ckpt", help="checkpoint location")
    parser.add_argument("--enableW8A8", action="store_true", help="apply W8A8 model")
    parser.add_argument('--outdir', default='./outputs', help='output dir')

    return parser.parse_args()


def get_point_map_estimation_opts():
    """Command-line arguments for point map estimation."""
    parser = argparse.ArgumentParser(
        "Test VGGT Point Map Estimation on ETH3d datasets.", 
        add_help=True
    )
    parser.add_argument("--ckpt", help="checkpoint location")
    parser.add_argument("--enableW8A8", action="store_true", help="apply W8A8 model")
    parser.add_argument("--dataset_dir", help="dataset location")
    parser.add_argument("--output_dir", type=str, default="", help="value for outdir")
    parser.add_argument("--use_proj", action="store_true")
    
    return parser.parse_args()


def get_pose_evaluation_opts():
    """Command-line arguments for CO3D pose evaluation (supports both single GPU and multi-GPU with SP)."""
    parser = argparse.ArgumentParser(
        description='Test VGGT Pose Evaluation on CO3D dataset.', 
        add_help=True
    )
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode (only test on apple category)')
    parser.add_argument('--min_num_images', type=int, default=50, 
                       help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, 
                       help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str, required=True, 
                       help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=True, 
                       help='Path to CO3D annotations')
    parser.add_argument("--ckpt", help="checkpoint location")
    parser.add_argument("--enableW8A8", action="store_true", 
                       help="apply W8A8 model")
    
    parser.add_argument('--use_sp', action='store_true',
                       help='Enable sequence parallel (use with torchrun for multi-GPU)')
    parser.add_argument('--ulysses_degree', type=int, default=1,
                       help='Ulysses parallelism degree (head-wise parallelism)')
    parser.add_argument('--ring_degree', type=int, default=1,
                       help='Ring attention degree (sequence-wise parallelism)')
    
    return parser.parse_args()


def get_pose_evaluation_opts_sp():
    """
    Command-line arguments for CO3D pose evaluation (multi-GPU with SP support).
    
    Note: This function is deprecated. Use get_pose_evaluation_opts() instead,
    which now supports both single GPU and multi-GPU modes.
    """
    parser = argparse.ArgumentParser(
        description='Multi-GPU CO3D Evaluation with SP support',
        add_help=True
    )
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (only test on apple category)')
    parser.add_argument('--min_num_images', type=int, default=50,
                       help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str, required=True,
                       help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=True,
                       help='Path to CO3D annotations')
    parser.add_argument("--ckpt", required=True,
                       help="Checkpoint location")
    parser.add_argument("--enableW8A8", action="store_true",
                       help="Apply W8A8 model")
    
    parser.add_argument("--enable_sp", action="store_true", 
                       help="Enable sequence parallel")
    parser.add_argument("--ulysses_degree", type=int, default=2,
                       help="Ulysses parallelism degree")
    parser.add_argument("--ring_degree", type=int, default=2,
                       help="Ring attention degree")
    
    return parser.parse_args()


def sync_and_get_time(start_time=None, use_syn=True, log_result=False, num_images=None):
    """Synchronize NPU and get timestamp."""
    if use_syn:
        torch.npu.synchronize()
    timestamp = time.time()
    if start_time is not None:
        timestamp -= start_time
        if log_result:
            if num_images is not None:
                logging.info(f"VGGT inference time cost is: {timestamp*1000:.2f} ms (num_images: {num_images})")
            else:
                logging.info(f"VGGT inference time cost is: {timestamp*1000:.2f} ms")
        return timestamp
    return timestamp