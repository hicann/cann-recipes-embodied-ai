# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/inference.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import datetime
import json
import logging
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import src.utils.misc as misc
from engine_tools import build_model, load_model
from src.utils.parser import get_args_parser
from src.dataset.constants import DATASET_DICT
from src.dataset.data_utils import to_batch_tensor
from src.dataset.datasets import SingleSequenceDataset
from src.utils.logging import setup_logging
from src.visualization.video_maker import make_video
from tools.lseg_feat_extractor import LSegFeatureExtractor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LOGGER = logging.getLogger("PerceptualModel")


def setup_experiment(args):
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    video_dir = os.path.join(log_dir, "videos")
    args.log_dir, args.ckpt_dir, args.video_dir = log_dir, checkpoint_dir, video_dir

    device = torch.device(args.device)
    misc.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    return device, log_dir


def setup_logger_and_save_args(args, log_dir):
    setup_logging(output=log_dir, level=logging.INFO)
    LOGGER.info(f"hostname: {os.uname().nodename}\n")
    LOGGER.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    LOGGER.info(f"Logging to {log_dir}")
    LOGGER.info(json.dumps(args.__dict__, indent=4, sort_keys=True))

    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)


def validate_dataset(args):
    if len(args.dataset) != 1:
        raise ValueError(f"Only one dataset is supported per inference session, "
                         f"but got {len(args.dataset)} datasets: {args.dataset}"
                         )


def resolve_annotation_path(args, annotation_file, is_val=False):
    if annotation_file is None:
        return None

    if "nuscenes" in args.dataset:
        return (
            "data/dataset_scene_list/nuscenes_val.txt"
            if is_val
            else "data/dataset_scene_list/nuscenes_train.txt"
        )

    annotation_path = f"{args.data_root}/{annotation_file}"
    if is_val and not os.path.exists(annotation_path):
        return None
    return annotation_path


def get_dataset_config(args):
    dataset_meta = DATASET_DICT[args.dataset[0]]

    train_annotation = resolve_annotation_path(args, dataset_meta["annotation_txt_file_train"])
    val_annotation = resolve_annotation_path(args, dataset_meta["annotation_txt_file_val"], is_val=True)

    num_context_timesteps = dataset_meta["num_context_timesteps"]
    num_target_timesteps = dataset_meta["num_target_timesteps"]

    if args.overwrite_train_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_train_ctx_view_with
    if args.overwrite_test_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_test_ctx_view_with
    if args.overwrite_train_tgt_view_with is not None:
        num_target_timesteps = args.overwrite_train_tgt_view_with

    return (
        train_annotation,
        val_annotation,
        num_context_timesteps,
        num_target_timesteps,
    )


def build_dataset(args, train_annotation, num_context_timesteps, num_target_timesteps):
    dataset = SingleSequenceDataset(
        data_root=args.data_root,
        annotation_txt_file_list=train_annotation,
        target_size=args.input_size,
        num_context_timesteps=num_context_timesteps,
        num_target_timesteps=num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        load_semantic_label=args.load_semantic_label,
        online_feat=args.online_feat,
        img_norm_for_online_feat=args.img_norm_for_online_feat,
    )

    LOGGER.info(f"Dataset contains {len(dataset):,} sequences using {train_annotation}.")
    return dataset


def build_feature_extractor(args):
    if not args.online_feat:
        return None

    LOGGER.info("Using online feature, loading feature extractor.")

    feat_extractor = LSegFeatureExtractor(
        args.lseg_model_pretrained_path,
        args.lseg_model_scratch_path,
        dtype=(
            torch.float16
            if os.environ.get("DISABLE_BFLOAT")
            else torch.bfloat16
        ),
    )

    LOGGER.info("Feature extractor loaded.")
    return feat_extractor


def get_scene_index(args, train_annotation, scene_id):
    with open(train_annotation, "r", encoding="utf-8") as f:
        lines_a = [
            line.strip().split("/")[-1].replace(".json", "")
            for line in f.readlines()
        ]

    if "waymo" not in args.dataset:
        return scene_id

    with open(f"{args.data_root}/scene_list/waymo_train_1.txt", "r", encoding="utf-8") as f:
        lines_b = [line.strip().split("/")[-1][:-5] for line in f.readlines()]

    name_to_index_a = {
        name: idx for idx, name in enumerate(lines_a)
    }

    name = lines_b[scene_id]
    return name_to_index_a[name]


def load_scene_data(dataset, scene_id_idx, scene_start_index, scene_end_index):
    LOGGER.info("Preparing data... (This may take a while)")

    data_dict_list = dataset.__getitem__(
        index=scene_id_idx,
        start_index=scene_start_index,
        end_index=scene_end_index,
    )
    data_dict_list = to_batch_tensor(data_dict_list)

    LOGGER.info("Done preparing data.")
    return data_dict_list


def generate_video_frames(data_dict_list, model, device, feat_extractor, output_name):
    video_frames_all = []

    for data_dict in data_dict_list:
        video_frames = make_video(
            dataset=None,
            model=model,
            device=device,
            output_filename=output_name,
            data_dict=data_dict,
            feat_extractor=feat_extractor,
            reverse_video=False,
            save_video=False,
        )
        video_frames_all.extend(video_frames)

    return video_frames_all, data_dict


def main(args):
    device, log_dir = setup_experiment(args)
    setup_logger_and_save_args(args, log_dir)

    validate_dataset(args)

    (
        train_annotation,
        val_annotation,
        num_context_timesteps,
        num_target_timesteps,
    ) = get_dataset_config(args)

    LOGGER.info(f"Dataset: {args.dataset}")
    LOGGER.info(f"annotation_txt_file_list_train: {train_annotation}")

    model = build_model(args)
    model = load_model(args, device, model)

    dataset = build_dataset(args, train_annotation, num_context_timesteps, num_target_timesteps, )
    feat_extractor = build_feature_extractor(args)

    scene_id = args.scene_id
    scene_start_index = args.scene_start_index
    scene_end_index = args.scene_end_index

    LOGGER.info(f"The id of the inference scene is {scene_id}, the starting frame is {scene_start_index}, "
                f"and the ending frame is {scene_end_index}")
    scene_id_idx = get_scene_index(args, train_annotation, scene_id, )
    data_dict_list = load_scene_data(dataset, scene_id_idx, scene_start_index, scene_end_index)

    output_name = (f"dataset_{args.dataset[0]}_scene_id_{scene_id}_"
                   f"{scene_start_index}-{scene_end_index}.mp4")

    video_frames_all, data_dict = generate_video_frames(data_dict_list, model, device, feat_extractor, output_name)

    LOGGER.info(f"Saved video to {output_name}")
    imageio.mimsave(output_name, video_frames_all, fps=data_dict["fps"])


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--scene_id", type=int, default=365)
    parser.add_argument("--scene_start_index", type=int, default=0)
    parser.add_argument("--scene_end_index", type=int, default=180)
    parser.add_argument("--overwrite_train_ctx_view_with", default=None, type=int)
    parser.add_argument("--overwrite_train_tgt_view_with", default=None, type=int)
    parser.add_argument("--overwrite_test_ctx_view_with", default=None, type=int)
    main(parser.parse_args())
