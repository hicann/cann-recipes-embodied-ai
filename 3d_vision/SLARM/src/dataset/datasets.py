# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/storm/dataset/storm_dataset.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import trange

from .constants import (
    DATASET_DICT,
    DATASETS,
    MEAN,
    STD,
    SEMANTIC_ID_TO_IDX_DICT,
)
from .data_utils import depth2xyz, resize_depth, resize_flow, to_float_tensor, to_tensor

logger = logging.getLogger("PerceptualModel")


class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets, dataset_names=None, per_dataset_sample_nums: dict = None):
        """
        Extended concatenated dataset class that tracks sub-dataset names
        and supports sample size configuration by name.

        Supports two scenarios:
        1. With `per_dataset_sample_nums`: Allocate contiguous sequence ranges
           based on specified sample sizes. Out-of-length mapping is handled
           by the sub-dataset's own modulo logic (enables repeated sampling).
        2. Without `per_dataset_sample_nums`: Allocate sequence ranges based on
           original dataset lengths (1:1 index mapping).

        Args:
            datasets: List of sub-datasets to concatenate.
            dataset_names: Optional list of names corresponding to sub-datasets.
                If None (default), sub-datasets will be named as
                "dataset_0", "dataset_1", ..., "dataset_N-1".
            per_dataset_sample_nums: Optional dictionary specifying sample sizes
                for each sub-dataset (key: dataset name, value: sample count).
                Takes precedence over default dataset lengths when provided;
                must contain all sub-dataset names.
        """
        super().__init__(datasets)

        if dataset_names is None:
            self.dataset_names = [f"dataset_{i}" for i, _ in enumerate(datasets)]
        else:
            if len(datasets) != len(dataset_names):
                raise ValueError("datasets and dataset_names must have the same length")
            self.dataset_names = dataset_names
        self.name_to_idx = {name: idx for idx, name in enumerate(self.dataset_names)}

        # Original data basic information
        self.sub_real_lens = [len(ds) for ds in self.datasets]  # Original length of each sub-dataset
        self.sub_real_offsets = [0]  # Global offsets for original data
        for ds_len in self.sub_real_lens[:-1]:
            self.sub_real_offsets.append(self.sub_real_offsets[-1] + ds_len)

        self.has_specified_sample_nums = per_dataset_sample_nums is not None
        if self.has_specified_sample_nums:
            self.per_dataset_sample_nums = per_dataset_sample_nums
            self._validate_sample_nums(per_dataset_sample_nums)
            self.sub_seq_ranges, self.total_seq_len = self._compute_seq_ranges_specified(
                per_dataset_sample_nums
            )
        else:
            self.sub_seq_ranges, self.total_seq_len = self._compute_seq_ranges_default()

    def __getitem__(self, seq_idx: int, *args, **kwargs):
        """Map global sequence index to a sample from the corresponding sub-dataset."""
        seq_idx = seq_idx % self.total_seq_len

        # Find the dataset that the sequence index belongs to
        dataset_name = None
        for name, (start, end) in self.sub_seq_ranges.items():
            if start <= seq_idx <= end:
                dataset_name = name
                break
        if dataset_name is None:
            raise RuntimeError(f"Sequence index {seq_idx} does not belong to any dataset")

        # Calculate local index (no modulo operation here; handled by sub-dataset itself)
        ds_idx = self.name_to_idx[dataset_name]
        if self.has_specified_sample_nums:
            seq_offset = seq_idx - self.sub_seq_ranges[dataset_name][0]
            local_idx = seq_offset  # Modulo is handled by the sub-dataset
        else:
            local_idx = seq_idx - self.sub_real_offsets[ds_idx]

        subset = self.datasets[ds_idx]
        return subset.__getitem__(local_idx, *args, **kwargs)

    def __len__(self) -> int:
        """Return total sequence length: sum of specified sample sizes (if provided)
        or sum of original dataset lengths."""
        return self.total_seq_len

    def _validate_sample_nums(self, per_dataset_sample_nums: dict):
        """Validate sample numbers for each sub-dataset."""
        missing_names = [name for name in per_dataset_sample_nums.keys() if name not in self.name_to_idx]
        if missing_names:
            raise ValueError(f"Dataset names {missing_names} not found!")
        for name, sample_num in per_dataset_sample_nums.items():
            if not isinstance(sample_num, int) or sample_num <= 0:
                raise ValueError(f"Sample num for {name} must be a positive integer!")
            ds_idx = self.name_to_idx[name]
            ds_len = self.sub_real_lens[ds_idx]
            if sample_num % ds_len != 0:
                raise ValueError(
                    f"Sample num for dataset '{name}' (={sample_num}) must be a multiple "
                    f"of its length (={ds_len})! Current: {sample_num} % {ds_len} = "
                    f"{sample_num % ds_len} (not 0)."
                )

    def _compute_seq_ranges_specified(self, per_dataset_sample_nums: dict) -> tuple:
        """Calculate sequence ranges based on specified sample sizes."""
        sub_seq_ranges = {}
        current_start = 0
        for name in self.dataset_names:
            if name not in per_dataset_sample_nums:
                raise ValueError(f"Dataset name {name} not found in per_dataset_sample_nums")
            sample_num = per_dataset_sample_nums[name]
            sub_seq_ranges[name] = (current_start, current_start + sample_num - 1)
            current_start += sample_num
        return sub_seq_ranges, current_start

    def _compute_seq_ranges_default(self) -> tuple:
        """Calculate sequence ranges based on original dataset lengths."""
        sub_seq_ranges = {}
        for idx, name in enumerate(self.dataset_names):
            start = self.sub_real_offsets[idx]
            end = self.sub_real_offsets[idx] + self.sub_real_lens[idx] - 1
            sub_seq_ranges[name] = (start, end)
        total_seq_len = sum(self.sub_real_lens)
        return sub_seq_ranges, total_seq_len


@dataclass
class CameraFrameContext:
    """Context for processing a single camera frame."""
    dataset_name: str
    camera: str
    frame_idx: int
    img_path: str
    normalized_intrinsics: Dict[str, Any]
    cam_to_world: Dict[str, Any]
    world_to_canonical: np.ndarray
    scene_json: Dict[str, Any]


@dataclass
class FrameData:
    """Container for frame-level tensor lists collected across cameras."""
    images: List[torch.Tensor]
    images_to_extract_feat: List[torch.Tensor]
    depths: List[torch.Tensor]
    pts3ds: List[torch.Tensor]
    valid_masks: List[torch.Tensor]
    sky_masks: List[torch.Tensor]
    flows: List[torch.Tensor]
    dynamic_masks: List[torch.Tensor]
    camtoworlds: List[torch.Tensor]
    intrinsics: List[torch.Tensor]
    ground_masks: List[torch.Tensor]
    semantic_labels: List[torch.Tensor]
    semantic_labels_mask: List[torch.Tensor]
    pseudo_depths: List[torch.Tensor]
    pseudo_depth_confs: List[torch.Tensor]
    frame_idx: int


@dataclass
class CameraVisualData:
    """Container for single-camera visual data values."""
    img: torch.Tensor
    img_to_extract_feat: Optional[torch.Tensor]
    sky: Optional[torch.Tensor]
    dynamic_mask: Optional[torch.Tensor]
    ground_mask: Optional[torch.Tensor]
    semantic: Optional[torch.Tensor]
    has_semantic: Optional[torch.Tensor]


@dataclass
class DatasetAttributes:
    """Container for dataset configuration attributes."""
    data_root: str
    target_size: Tuple[int, int]
    num_context_timesteps: int
    num_target_timesteps: int
    num_max_cams: int
    timespan: float
    load_depth: bool
    load_flow: bool
    load_pseudo_depth: bool
    load_dynamic_mask: bool
    load_ground_label: bool
    load_semantic_label: bool
    skip_sky_mask: bool
    online_feat: bool
    equispaced: bool
    return_context_as_target: bool
    only_interp: bool


@dataclass
class SceneFrameContext:
    """Scene-level context shared across cameras in a single frame."""
    scene_json: Dict[str, Any]
    dataset_name: str
    frame_idx: int
    normalized_intrinsics: Dict[str, Any]
    cam_to_world: Dict[str, Any]
    world_to_canonical: np.ndarray


class PerceptualModelDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: Union[str, List[str]],
        target_size: Tuple[int, int] = (160, 240),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,  # 2.0 seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_pseudo_depth: bool = False,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        load_semantic_label: bool = False,
        return_context_as_target: bool = False,
        skip_sky_mask: bool = False,
        feat_input_size: Tuple[int, int] = (320, 480),
        online_feat: bool = False,
        img_norm_for_online_feat: bool = False,
        only_interp: bool = True,
    ):
        super().__init__()
        attributes = DatasetAttributes(
            data_root=data_root,
            target_size=target_size,
            num_context_timesteps=num_context_timesteps,
            num_target_timesteps=num_target_timesteps,
            num_max_cams=num_max_cams,
            timespan=timespan,
            load_depth=load_depth,
            load_flow=load_flow,
            load_pseudo_depth=load_pseudo_depth,
            load_dynamic_mask=load_dynamic_mask,
            load_ground_label=load_ground_label,
            load_semantic_label=load_semantic_label,
            skip_sky_mask=skip_sky_mask,
            online_feat=online_feat,
            equispaced=equispaced,
            return_context_as_target=return_context_as_target,
            only_interp=only_interp,
        )
        self._init_attributes(attributes)

        self.annotations = self._load_annotations(
            annotation_txt_file_list, data_root, subset_indices, num_replicas
        )

        self.num_replicas = num_replicas

        self._init_image_transformations(target_size, feat_input_size, img_norm_for_online_feat)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(
        self, index: int, context_frame_idx: int = -1, return_all=False
    ) -> Dict[str, Any]:
        try:
            return self.get_segment(index, context_frame_idx, return_all)
        except Exception as e:
            scene_json = self.annotations[index % len(self.annotations)]
            scene_id = scene_json["scene_id"]
            logger.info(
                f"Error in scene_id: {scene_id}, "
                f"context_frame_idx: {context_frame_idx}, "
                f"scene_name: {scene_json['scene_name']}"
            )
            logger.info(e)
            try:
                return self.get_segment(index + 1, 0, return_all=return_all)
            except Exception as inner_e:
                logger.info(inner_e)
                return self.get_segment(index + 1, return_all=return_all)

    @staticmethod
    def _stack_optional(tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Stack tensors if list is non-empty, else return None."""
        if len(tensor_list) > 0:
            return torch.stack(tensor_list)
        return None

    def get_interval(self, fps):
        # The evaluation/inference strategy is determined based on the training strategy.
        num_max_future_frames = int(self.timespan * fps)
        max_idx = np.arange(
                0,
                num_max_future_frames,
                num_max_future_frames // self.num_context_timesteps,
        ).max()
        if self.equispaced and self.only_interp:
            interval = max_idx + 1
        else:
            interval = num_max_future_frames
        return interval

    def get_frame(
        self,
        scene_json: Dict[str, Any],
        frame_idx: int,
        source_frame_idx: int = -1,
    ) -> Dict[str, Any]:
        """Retrieve a single frame from the dataset."""
        normalized_intrinsics = scene_json["normalized_intrinsics"]
        dataset_name = scene_json["dataset"]
        cam_to_world = scene_json["camera_to_world"]

        images, depths, sky_masks, flows = [], [], [], []
        pseudo_depths, pseudo_depth_confs, images_to_extract_feat = [], [], []
        pts3ds, valid_masks, camtoworlds, intrinsics = [], [], [], []
        dynamic_masks, ground_masks, semantic_labels, semantic_labels_mask = [], [], [], []

        frame_data = FrameData(
            images=images, images_to_extract_feat=images_to_extract_feat,
            depths=depths, pts3ds=pts3ds, valid_masks=valid_masks,
            sky_masks=sky_masks, flows=flows, dynamic_masks=dynamic_masks,
            camtoworlds=camtoworlds, intrinsics=intrinsics,
            ground_masks=ground_masks, semantic_labels=semantic_labels,
            semantic_labels_mask=semantic_labels_mask,
            pseudo_depths=pseudo_depths, pseudo_depth_confs=pseudo_depth_confs,
            frame_idx=frame_idx,
        )

        if source_frame_idx < 0:
            source_frame_idx = frame_idx

        camera_list = DATASET_DICT[dataset_name]["camera_list"][self.num_max_cams]
        ref_camera_name = DATASET_DICT[dataset_name]["ref_camera"]

        world_to_canonical = np.linalg.inv(torch.tensor(cam_to_world[ref_camera_name][source_frame_idx]))

        scene_ctx = SceneFrameContext(
            scene_json=scene_json,
            dataset_name=dataset_name,
            frame_idx=frame_idx,
            normalized_intrinsics=normalized_intrinsics,
            cam_to_world=cam_to_world,
            world_to_canonical=world_to_canonical,
        )
        for camera in camera_list:
            self._process_single_camera(camera, scene_ctx, frame_data)

        return self._stack_and_build_data_dict(frame_data)

    def get_segment(
        self, index: int, context_frame_idx: int = -1, return_all=False
    ) -> Dict[str, Any]:
        """Retrieve a segment of frames from the dataset."""
        scene_json = self.annotations[index % len(self.annotations)]
        scene_id = scene_json["scene_id"]
        num_timesteps = scene_json["num_timesteps"]
        fps = scene_json["fps"]
        num_max_future_frames = int(self.timespan * fps)
        if num_max_future_frames > num_timesteps:
            num_max_future_frames = int(fps)
        time_in_seconds = scene_json["normalized_time"]
        interval = self.get_interval(fps)

        if context_frame_idx < 0 or context_frame_idx + interval > num_timesteps:
            context_frame_idx = np.random.randint(0, num_timesteps - interval + 1)
        if context_frame_idx + interval > num_timesteps:
            raise ValueError(
                f"scene_id: {scene_id}, context_frame_idx: {context_frame_idx}, "
                f"num_timesteps: {num_timesteps}, num_max_future_frames: {num_max_future_frames}"
            )

        context_frame_idxs, target_frame_idxs = self._compute_frame_indices(
            context_frame_idx, num_timesteps, fps, num_max_future_frames, return_all
        )

        context_dict_list = self._load_frames_dict_list(
            scene_json, context_frame_idxs, context_frame_idxs[0], time_in_seconds
        )

        if self.return_context_as_target:
            target_frame_idxs = context_frame_idxs

        target_dict_list = self._load_frames_dict_list(
            scene_json, target_frame_idxs, context_frame_idxs[0], time_in_seconds
        )

        context_dict = self._collate_frames_dict(context_dict_list)
        target_dict = self._collate_frames_dict(target_dict_list)

        return self._prepare_segment_sample(
            context_dict, target_dict, scene_json, scene_id, fps
        )

    def get_one_scene(
        self, index: int, start_index=0, end_index=60, time_step: int = 5, return_all=True
    ) -> Dict[str, Any]:
        """Retrieve a segment of frames from a single scene."""
        try:
            scene_json = self.annotations[index % len(self.annotations)]
            scene_id = scene_json["scene_id"]
            num_timesteps = scene_json["num_timesteps"]
            fps = scene_json["fps"]
            time_in_seconds = scene_json["normalized_time"]
            end_index = min(end_index, num_timesteps - 1)

            context_frame_idxs, target_frame_idxs = self._compute_one_scene_frame_indices(
                start_index, end_index, time_step, fps, return_all
            )

            context_dict_list = self._load_frames_dict_list(
                scene_json, context_frame_idxs, context_frame_idxs[0], time_in_seconds
            )

            if self.return_context_as_target:
                target_frame_idxs = context_frame_idxs

            target_dict_list = self._load_frames_dict_list(
                scene_json, target_frame_idxs, context_frame_idxs[0], time_in_seconds
            )

            context_dict = self._collate_frames_dict(context_dict_list)
            target_dict = self._collate_frames_dict(target_dict_list)

            sample = {
                "context": context_dict,
                "target": target_dict,
                "scene_id": scene_id,
                "scene_name": scene_json["scene_name"],
                "width": self.target_size[1],
                "height": self.target_size[0],
                "fps": fps,
                "timespan": self.timespan,
                "num_max_cams": self.num_max_cams,
            }
            return to_float_tensor(sample)
        except Exception as e:
            logger.info(
                f"Error in scene_id: {scene_id}, "
                f"context_frame_idx: {context_frame_idxs}, "
                f"scene_name: {scene_json['scene_name']}"
            )
            logger.info(e)
            return None

    def _init_attributes(self, attributes: DatasetAttributes):
        """Initialize basic dataset attributes."""
        self.data_root = attributes.data_root
        self.target_size = attributes.target_size
        self.num_context_timesteps = attributes.num_context_timesteps
        self.num_target_timesteps = attributes.num_target_timesteps
        self.num_max_cams = attributes.num_max_cams
        self.timespan = attributes.timespan
        self.load_depth = attributes.load_depth
        self.load_flow = attributes.load_flow
        self.load_pseudo_depth = attributes.load_pseudo_depth
        self.load_dynamic_mask = attributes.load_dynamic_mask
        self.load_ground_label = attributes.load_ground_label
        self.load_semantic_label = attributes.load_semantic_label
        self.skip_sky_mask = attributes.skip_sky_mask
        self.online_feat = attributes.online_feat
        self.equispaced = attributes.equispaced
        self.return_context_as_target = attributes.return_context_as_target
        self.only_interp = attributes.only_interp

    def _process_single_camera(
        self,
        camera: str,
        scene_ctx: SceneFrameContext,
        frame_data: FrameData,
    ):
        """Process visual and geometry data for a single camera."""
        img_relative_path = scene_ctx.scene_json["relative_image_path"][camera][scene_ctx.frame_idx]
        if scene_ctx.dataset_name in ["waymo", "nuscenes"]:
            img_relative_path = img_relative_path.replace("images", f"images_4")
            img_relative_path = img_relative_path.replace("sweeps", f"sweeps_4")
            img_relative_path = img_relative_path.replace("samples", f"samples_4")

        img_path = os.path.join(self.data_root, "datasets", scene_ctx.dataset_name, img_relative_path)

        visual_data = self._load_camera_visual_data(scene_ctx.dataset_name, img_path, camera)
        frame_data.images.append(visual_data.img)
        self._append_visual_data(visual_data, frame_data)

        ctx = CameraFrameContext(
            scene_ctx.dataset_name, camera, scene_ctx.frame_idx, img_path,
            scene_ctx.normalized_intrinsics, scene_ctx.cam_to_world,
            scene_ctx.world_to_canonical, scene_ctx.scene_json,
        )
        geo = self._load_camera_geometry_data(ctx, img_path)
        self._append_geometry_data(geo, frame_data)

    def _append_visual_data(
        self,
        visual_data: CameraVisualData,
        frame_data: FrameData,
    ):
        """Append visual data to the provided lists if not None."""
        if visual_data.img_to_extract_feat is not None:
            frame_data.images_to_extract_feat.append(visual_data.img_to_extract_feat)
        if visual_data.sky is not None:
            frame_data.sky_masks.append(visual_data.sky)
        if visual_data.dynamic_mask is not None:
            frame_data.dynamic_masks.append(visual_data.dynamic_mask)
        if visual_data.ground_mask is not None:
            frame_data.ground_masks.append(visual_data.ground_mask)
        if visual_data.semantic is not None:
            frame_data.semantic_labels.append(visual_data.semantic)
            frame_data.semantic_labels_mask.append(visual_data.has_semantic)

    def _load_camera_geometry_data(
        self,
        ctx: CameraFrameContext,
        img_path: str,
    ) -> Dict[str, Any]:
        """Load camera pose, depth/flow, and pseudo depth for a single camera."""
        camtoworld, intrinsic = self._process_camera_pose(ctx)
        depth, pts3d, valid_mask, flow = self._load_depth_and_flow(ctx, camtoworld)
        pseudo_depth, pseudo_depth_conf = self._load_pseudo_depth(ctx.dataset_name, img_path)
        return {
            "camtoworld": camtoworld,
            "intrinsic": intrinsic,
            "depth": depth,
            "pts3d": pts3d,
            "valid_mask": valid_mask,
            "flow": flow,
            "pseudo_depth": pseudo_depth,
            "pseudo_depth_conf": pseudo_depth_conf,
        }

    def _append_geometry_data(
        self,
        geo: Dict[str, Any],
        frame_data: FrameData,
    ):
        """Append geometry data to the provided lists."""
        frame_data.camtoworlds.append(geo["camtoworld"])
        frame_data.intrinsics.append(geo["intrinsic"])
        if geo["depth"] is not None:
            frame_data.depths.append(geo["depth"])
        if geo["pts3d"] is not None:
            frame_data.pts3ds.append(geo["pts3d"])
        if geo["valid_mask"] is not None:
            frame_data.valid_masks.append(geo["valid_mask"])
        if geo["flow"] is not None:
            frame_data.flows.append(geo["flow"])
        if geo["pseudo_depth"] is not None:
            frame_data.pseudo_depths.append(geo["pseudo_depth"])
        if geo["pseudo_depth_conf"] is not None:
            frame_data.pseudo_depth_confs.append(geo["pseudo_depth_conf"])

    def _load_camera_visual_data(
        self,
        dataset_name: str,
        img_path: str,
        camera: str,
    ) -> CameraVisualData:
        """Load RGB image, semantic image, and masks for a single camera."""
        img, img_to_extract_feat = self._load_rgb_image(dataset_name, img_path, camera)
        semantic_image_np = self._load_semantic_image_if_needed(dataset_name, img_path)
        sky = self._load_sky_mask(dataset_name, img_path, camera, semantic_image_np)
        dynamic_mask = self._load_dynamic_mask(dataset_name, img_path, semantic_image_np)
        ground_mask = self._load_ground_mask(dataset_name, img_path, semantic_image_np)
        semantic, has_semantic = self._load_semantic_label(dataset_name, img_path)
        return CameraVisualData(
            img=img, img_to_extract_feat=img_to_extract_feat, sky=sky,
            dynamic_mask=dynamic_mask, ground_mask=ground_mask,
            semantic=semantic, has_semantic=has_semantic,
        )

    def _load_annotations(
        self,
        annotation_txt_file_list: Union[str, List[str]],
        data_root: str,
        subset_indices: Optional[List[int]],
        num_replicas: int,
    ) -> List[Dict[str, Any]]:
        """Load annotation files and apply subset and replication."""
        if isinstance(annotation_txt_file_list, str):
            annotation_txt_file_list = [annotation_txt_file_list]

        # Read all lines from annotation_txt_file_list (scene path/JSON)
        annotation_paths = []
        for annotation_txt_file in annotation_txt_file_list:
            with open(annotation_txt_file, "r") as f:
                annotation_paths += [line.strip() for line in f.readlines() if line.strip()]

        if subset_indices is not None:
            annotation_paths = [annotation_paths[i] for i in subset_indices]

        annotations = []
        for annotation_path in annotation_paths:
            with open(os.path.join(data_root, annotation_path), "r") as f:
                annotations.append(json.load(f))
        logger.info(f"Loaded {len(annotations)} annotations.")

        if num_replicas > 1:
            annotations *= num_replicas

        return annotations

    def _init_image_transformations(
        self,
        target_size: Tuple[int, int],
        feat_input_size: Tuple[int, int],
        img_norm_for_online_feat: bool,
    ):
        """Initialize image transformation pipelines."""
        self.img_transformation = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
            ]
        )
        # original size transformation for online feature extraction
        if img_norm_for_online_feat:
            img_to_extract_feat_transformation_list = [
                transforms.Resize(feat_input_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
                # Larger deviation from offline features;
                # normalization decision pending implementation results
            ]
        else:
            img_to_extract_feat_transformation_list = [
                transforms.Resize(feat_input_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
            ]
        self.img_to_extract_feat_transformation = transforms.Compose(img_to_extract_feat_transformation_list)

    def _load_rgb_image(
        self,
        dataset_name: str,
        img_path: str,
        camera: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load RGB image and online feature extraction image."""
        raw_img = Image.open(img_path).convert("RGB")
        img = self.img_transformation(raw_img)

        img_to_extract_feat = None
        if self.online_feat:
            if dataset_name == "driving_sim" or dataset_name == "b2d":
                img_to_extract_feat = self.img_to_extract_feat_transformation(raw_img)
            else:
                img_name = img_path.split('/')[-2].strip()
                img_to_extract_feat_path = img_path.replace(img_name, "images")
                img_to_extract_feat = Image.open(img_to_extract_feat_path).convert("RGB")
                img_to_extract_feat = self.img_to_extract_feat_transformation(img_to_extract_feat)

        return img, img_to_extract_feat

    def _load_semantic_image_if_needed(
        self,
        dataset_name: str,
        img_path: str,
    ) -> Optional[np.ndarray]:
        """Load semantic image if needed for sky/dynamic/ground masks."""
        is_b2d_with_features = (
            dataset_name == "b2d"
            and (not self.skip_sky_mask or self.load_dynamic_mask or self.load_ground_label)
        )
        if is_b2d_with_features:
            semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
            semantic_image = Image.open(semantic_path).resize(self.target_size[::-1], resample=Image.NEAREST)
            semantic_image_np = np.array(semantic_image)
            if len(semantic_image_np.shape) > 2:
                semantic_image_np = semantic_image_np[..., 0]
            return semantic_image_np

        if dataset_name == "driving_sim" and (self.load_dynamic_mask or self.load_ground_label):
            semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
            semantic_image_np = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            semantic_image_np = cv2.resize(
                semantic_image_np, self.target_size[::-1], interpolation=cv2.INTER_NEAREST
            )
            return semantic_image_np

        return None

    def _load_sky_mask(
        self,
        dataset_name: str,
        img_path: str,
        camera: str,
        semantic_image_np: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """Load sky mask."""
        if dataset_name in ["waymo", "nuscenes", "driving_sim"]:
            if dataset_name == "nuscenes":
                sky_path = img_path.replace("samples", "samples_sky_mask")
                sky_path = sky_path.replace("sweeps", "sweeps_sky_mask")
            elif dataset_name == "waymo":
                sky_path = img_path.replace("images_4", "sky_masks")
            elif dataset_name == "driving_sim":
                sky_path = img_path.replace(f"rgb_{camera}", f"sky_mask_{camera}")
            sky_path = sky_path.replace("jpg", "png")
            if self.skip_sky_mask:
                sky = torch.zeros(self.target_size[0], self.target_size[1]).float()
            else:
                try:
                    new_sky_path = sky_path.replace("STORM2", "STORM_masks")
                    sky = Image.open(new_sky_path).convert("L").resize(self.target_size[::-1])
                except FileNotFoundError:
                    sky = Image.open(sky_path).convert("L").resize(self.target_size[::-1])
            sky = to_tensor(np.array(sky) > 0).float()
            return sky
        elif dataset_name == "b2d" and not self.skip_sky_mask:
            sky = to_tensor(semantic_image_np == 11).float()
            return sky
        return None

    def _load_dynamic_mask(
        self,
        dataset_name: str,
        img_path: str,
        semantic_image_np: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """Load dynamic mask for dynamic region evaluation."""
        if not self.load_dynamic_mask:
            return None

        if dataset_name == "waymo":
            dynamic_path = img_path.replace("images_8", "dynamic_masks")
            dynamic_path = dynamic_path.replace("images_4", "dynamic_masks")
            dynamic_path = dynamic_path.replace("jpg", "png")
            if not os.path.exists(dynamic_path):
                dynamic_path = dynamic_path.replace("STORM2", "STORM")
            dynamic_mask = Image.open(dynamic_path).convert("L").resize(self.target_size[::-1])
            dynamic_mask = to_tensor(np.array(dynamic_mask) > 0).float()
            return dynamic_mask
        elif dataset_name == "b2d":
            dynamic = to_tensor(semantic_image_np == 21).float()
            return dynamic
        elif dataset_name == "driving_sim":
            dynamic = to_tensor(semantic_image_np == 24).float()
            return dynamic
        return None

    def _load_ground_mask(
        self,
        dataset_name: str,
        img_path: str,
        semantic_image_np: Optional[np.ndarray],
    ) -> Optional[torch.Tensor]:
        """Load ground mask for flow evaluation (exclude ground lidar points)."""
        if not self.load_ground_label:
            return None

        if dataset_name == "waymo":
            ground_path = img_path.replace("images", "ground_label")
            ground_path = ground_path.replace("jpg", "png")
            ground = Image.open(ground_path).convert("L").resize(self.target_size[::-1])
            ground = to_tensor(np.array(ground) > 0).float()
            return ground
        elif dataset_name == "b2d":
            ground = to_tensor(semantic_image_np == 1).float()
            return ground
        elif dataset_name == "driving_sim":
            ground = to_tensor(semantic_image_np == 1).float()
            return ground
        return None

    def _load_semantic_label(
        self,
        dataset_name: str,
        img_path: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load semantic label with class remapping."""
        if not self.load_semantic_label:
            return None, None

        semantic_tensor = to_tensor(np.zeros(self.target_size)).float()
        has_semantic = torch.tensor(False)
        mapping = torch.arange(29, dtype=torch.int)

        if dataset_name == "waymo":
            semantic_path = img_path.replace("images", "semantic_segs")
            semantic_path = semantic_path.replace("jpg", "npy")
            if os.path.exists(semantic_path):
                semantic_np = np.load(semantic_path)
                semantic_np = cv2.resize(semantic_np, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
                semantic_tensor = to_tensor(semantic_np.astype(np.float32)).float()
                has_semantic = torch.tensor(True)
        elif dataset_name == "b2d":
            semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
            semantic_img = Image.open(semantic_path).resize(self.target_size[::-1], resample=Image.NEAREST)
            semantic_np = np.array(semantic_img)
            if len(semantic_np.shape) > 2:
                semantic_np = semantic_np[..., 0]
            semantic_tensor = to_tensor(semantic_np.astype(np.float32)).float()
            has_semantic = torch.tensor(True)
        elif dataset_name == "driving_sim":
            semantic_path = img_path.replace("rgb", "semantic").replace(".jpg", ".png")
            semantic_np = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            semantic_np = cv2.resize(semantic_np, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
            semantic_tensor = to_tensor(semantic_np.astype(np.float32)).float()
            has_semantic = torch.tensor(True)

        for key, value in SEMANTIC_ID_TO_IDX_DICT[dataset_name].items():
            mapping[key] = value
        semantic_tensor = mapping[semantic_tensor.to(torch.int32)]

        return semantic_tensor, has_semantic

    def _process_camera_pose(
        self,
        ctx: CameraFrameContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process camera pose and intrinsics."""
        camera2world = ctx.cam_to_world[ctx.camera][ctx.frame_idx]

        camtoworld = (
            DATASETS[ctx.dataset_name]["canonical_to_flu"]
            @ ctx.world_to_canonical
            @ camera2world
            @ DATASETS[ctx.dataset_name]["opencv2dataset"]
        )
        camtoworld = to_tensor(camtoworld)

        fx, fy, cx, cy = np.array(ctx.normalized_intrinsics[ctx.camera])
        fx = fx * self.target_size[1]
        fy = fy * self.target_size[0]
        cx = cx * self.target_size[1]
        cy = cy * self.target_size[0]
        intrinsics = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ]
        ).float()

        return camtoworld, intrinsics

    def _load_depth_and_flow(
        self,
        ctx: CameraFrameContext,
        camtoworld: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load depth and flow data, return (depth, pts3d, valid_mask, flow)."""
        if not self.load_depth and not self.load_flow:
            return None, None, None, None

        loaders = {
            "waymo": self._load_depth_flow_waymo,
            "nuscenes": self._load_depth_flow_nuscenes,
            "driving_sim": self._load_depth_flow_driving_sim,
            "b2d": self._load_depth_flow_b2d,
        }
        loader = loaders.get(ctx.dataset_name)
        if loader is None:
            return None, None, None, None
        return loader(ctx, camtoworld)

    def _load_depth_flow_waymo(
        self,
        ctx: CameraFrameContext,
        camtoworld: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load depth and flow for waymo dataset."""
        depth, pts3d, valid_mask, flow = None, None, None, None
        depth_path = ctx.img_path.replace("images", "depth_flows").replace("jpg", "npy")
        depth_and_flow = np.load(depth_path)
        if self.load_depth:
            depth = depth_and_flow[..., 0]
            depth = torch.tensor(depth).float()
            depth = resize_depth(depth, self.target_size)
            pts3d = depth2xyz(
                np.array(depth),
                fxfycxcy=np.array(ctx.normalized_intrinsics[ctx.camera]),
                cam2world=np.array(camtoworld),
                return_pixel=True
            )
            pts3d = torch.tensor(pts3d).float()
            valid_mask = depth > 0.0
        if self.load_flow:
            flow = depth_and_flow[..., 1:]
            flow = torch.tensor(flow).float()
            flow = resize_flow(flow, self.target_size)
            flow = (
                flow
                @ torch.tensor(
                    (
                        ctx.world_to_canonical
                        @ ctx.cam_to_world[ctx.camera][ctx.frame_idx]
                        @ np.linalg.inv(ctx.scene_json["camera_to_ego"][ctx.camera])
                    )
                )
                .float()[:3, :3]
                .T.contiguous()
            )
        return depth, pts3d, valid_mask, flow

    def _load_depth_flow_nuscenes(
        self,
        ctx: CameraFrameContext,
        camtoworld: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load depth for nuscenes dataset."""
        depth_path = ctx.img_path.replace("samples", "samples_depth")
        depth_path = depth_path.replace("sweeps", "sweeps_depth")
        depth_path = depth_path.replace("jpg", "npy")
        depth = np.load(depth_path)
        depth = torch.tensor(depth).float()
        depth = resize_depth(depth, self.target_size)
        return depth, None, None, None

    def _load_depth_flow_driving_sim(
        self,
        ctx: CameraFrameContext,
        camtoworld: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load depth and flow for driving_sim dataset."""
        depth, pts3d, valid_mask, flow = None, None, None, None
        if self.load_depth:
            depth_path = ctx.img_path.replace(f"rgb_{ctx.camera}", f"depth_{ctx.camera}")
            depth_path = depth_path.replace("jpg", "png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 100.
            depth = torch.tensor(depth).float()
            depth = resize_depth(depth, self.target_size)
            pts3d = depth2xyz(
                np.array(depth),
                fxfycxcy=np.array(ctx.normalized_intrinsics[ctx.camera]),
                cam2world=np.array(camtoworld),
                return_pixel=True
            )
            pts3d = torch.tensor(pts3d).float()
            valid_mask = (depth > 0.0) & (depth < 200)
        if self.load_flow:
            if ctx.frame_idx == 0:
                flow = torch.zeros((self.target_size[0], self.target_size[1], 3), dtype=torch.float32)
            else:
                flow_path = ctx.img_path.replace(f"rgb_{ctx.camera}", f"scene_flow_{ctx.camera}")
                flow_path = flow_path.replace("jpg", "png")
                flow_data = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
                flow = flow_data.astype(np.float32)
                flow = (flow / 65535 * 2 - 1) * 5
                flow = flow * 20
                flow = torch.tensor(flow).float()
                flow = resize_flow(flow, self.target_size)
                flow = (
                    flow
                    @ torch.tensor(
                        (
                            ctx.world_to_canonical
                            @ ctx.cam_to_world[ctx.camera][ctx.frame_idx]
                            @ np.linalg.inv(ctx.scene_json["camera_to_ego"][ctx.camera])
                        )
                    )
                    .float()[:3, :3]
                    .T.contiguous()
                )
        return depth, pts3d, valid_mask, flow

    def _load_depth_flow_b2d(
        self,
        ctx: CameraFrameContext,
        camtoworld: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load depth and flow for b2d dataset."""
        depth, pts3d, valid_mask, flow = None, None, None, None
        if self.load_depth:
            depth_path = ctx.img_path.replace("rgb", "depth").replace(".jpg", ".png")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.01
            depth[depth > 400] = 0.0
            depth = torch.tensor(depth).float()
            depth = resize_depth(depth, self.target_size)
            pts3d = depth2xyz(
                np.array(depth),
                fxfycxcy=np.array(ctx.normalized_intrinsics[ctx.camera]),
                cam2world=np.array(camtoworld),
                return_pixel=True
            )
            pts3d = torch.tensor(pts3d).float()
            valid_mask = (depth > 0.0) & (depth < 200)
        if self.load_flow:
            flow = torch.zeros((self.target_size[0], self.target_size[1], 3), dtype=torch.float32)
        return depth, pts3d, valid_mask, flow

    def _load_pseudo_depth(
        self,
        dataset_name: str,
        img_path: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load pseudo depth and its confidence."""
        if not (dataset_name == "waymo" and self.load_pseudo_depth):
            return None, None

        pseudo_depth_path = img_path.replace("images", "pseudo_depth").replace("jpg", "npy")
        pseudo_depth = np.load(pseudo_depth_path)
        pseudo_depth = torch.tensor(pseudo_depth).float()
        pseudo_depth = resize_depth(pseudo_depth, self.target_size)

        pseudo_depth_conf_path = img_path.replace("images", "pseudo_depth").replace(".jpg", "_conf.npy")
        pseudo_depth_conf = np.load(pseudo_depth_conf_path)
        pseudo_depth_conf = torch.tensor(pseudo_depth_conf).float()
        pseudo_depth_conf = resize_depth(pseudo_depth_conf, self.target_size)

        return pseudo_depth, pseudo_depth_conf

    def _stack_and_build_data_dict(
        self,
        frame_data: FrameData,
    ) -> Dict[str, Any]:
        """Stack tensors and build the final data dictionary."""
        frame_images = torch.stack(frame_data.images)
        frame_depths = self._stack_optional(frame_data.depths)
        frame_pts3ds = self._stack_optional(frame_data.pts3ds)
        frame_valid_masks = self._stack_optional(frame_data.valid_masks)
        frame_sky_masks = self._stack_optional(frame_data.sky_masks)
        frame_flows = self._stack_optional(frame_data.flows)
        frame_dynamic_masks = self._stack_optional(frame_data.dynamic_masks)
        frame_camtoworlds = torch.stack(frame_data.camtoworlds)
        frame_intrinsics = torch.stack(frame_data.intrinsics)
        ground_masks_tensor = self._stack_optional(frame_data.ground_masks)
        semantic_labels_tensor = self._stack_optional(frame_data.semantic_labels)
        semantic_labels_mask_tensor = self._stack_optional(frame_data.semantic_labels_mask)
        frame_pseudo_depths = self._stack_optional(frame_data.pseudo_depths)
        frame_pseudo_depth_confs = self._stack_optional(frame_data.pseudo_depth_confs)

        data_dict = {
            "image": frame_images,
            "camtoworld": frame_camtoworlds,
            "intrinsics": frame_intrinsics,
            "frame_idx": frame_data.frame_idx,
            "depth": frame_depths,
            "pts3d": frame_pts3ds,
            "valid_masks": frame_valid_masks,
            "sky_masks": frame_sky_masks,
            "flow": frame_flows,
            "dynamic_masks": frame_dynamic_masks,
            "ground_masks": ground_masks_tensor,
            "semantic_labels": semantic_labels_tensor,
            "semantic_labels_mask": semantic_labels_mask_tensor,
            "pseudo_depth": frame_pseudo_depths,
            "pseudo_depth_conf": frame_pseudo_depth_confs
        }

        if self.online_feat:
            frame_images_to_extract_feat = torch.stack(frame_data.images_to_extract_feat)
            data_dict["frame_images_to_extract_feat"] = frame_images_to_extract_feat

        return {k: v for k, v in data_dict.items() if v is not None}

    def _compute_frame_indices(
        self,
        context_frame_idx: int,
        num_timesteps: int,
        fps: float,
        num_max_future_frames: int,
        return_all: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute context and target frame indices based on sampling strategy."""
        interval = self.get_interval(fps)

        # Compute context frame indices
        if self.equispaced:
            context_frame_idxs = np.arange(
                context_frame_idx,
                context_frame_idx + num_max_future_frames,
                num_max_future_frames // self.num_context_timesteps,
            )
        else:
            context_frame_idxs = np.random.choice(
                np.arange(context_frame_idx, context_frame_idx + num_max_future_frames),
                size=self.num_context_timesteps,
                replace=False,
            )
            context_frame_idxs = sorted(context_frame_idxs)

        # Compute target frame indices
        if self.only_interp:
            target_frame_idxs = np.arange(
                context_frame_idxs[0], context_frame_idxs[-1] + 1
            )
        else:
            target_frame_idxs = np.arange(
                context_frame_idx, context_frame_idx + num_max_future_frames
            )

        if not return_all:
            target_frame_idxs = np.random.choice(
                target_frame_idxs, self.num_target_timesteps, replace=False
            )
        target_frame_idxs = [min(idx, num_timesteps - 1) for idx in target_frame_idxs]
        target_frame_idxs = sorted(target_frame_idxs)

        return context_frame_idxs, np.array(target_frame_idxs)

    def _load_frames_dict_list(
        self,
        scene_json: Dict[str, Any],
        frame_idxs: np.ndarray,
        source_frame_idx: int,
        time_in_seconds: List[float],
    ) -> List[Dict[str, Any]]:
        """Load frame data for a list of frame indices and add time information."""
        frames_dict_list = []
        for frame_idx in frame_idxs:
            frame_dict = self.get_frame(
                scene_json=scene_json,
                frame_idx=frame_idx,
                source_frame_idx=source_frame_idx,
            )
            frame_dict["time"] = torch.tensor(
                [time_in_seconds[frame_idx] - time_in_seconds[source_frame_idx]]
                * self.num_max_cams
            )
            frames_dict_list.append(frame_dict)
        return frames_dict_list

    def _collate_frames_dict(
        self, frames_dict_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate frame data and concatenate tensors along batch dimension."""
        frames_dict = default_collate(frames_dict_list)
        for k, v in frames_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                frames_dict[k] = torch.cat([d for d in v], dim=0)
        return frames_dict

    def _prepare_segment_sample(
        self,
        context_dict: Dict[str, Any],
        target_dict: Dict[str, Any],
        scene_json: Dict[str, Any],
        scene_id: int,
        fps: float,
    ) -> Dict[str, Any]:
        """Prepare the final sample dictionary with all metadata."""
        sample = {
            "context": context_dict,
            "target": target_dict,
            "dataset_name": scene_json['dataset'],
            "scene_id": scene_id,
            "scene_name": scene_json["scene_name"],
            "width": self.target_size[1],
            "height": self.target_size[0],
            "fps": fps,
            "timespan": self.timespan,
            "num_max_cams": self.num_max_cams,
        }
        return to_float_tensor(sample)

    def _compute_one_scene_frame_indices(
        self,
        start_index: int,
        end_index: int,
        time_step: int,
        fps: float,
        return_all: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute context and target frame indices for one scene."""
        context_frame_idxs = np.arange(start_index, end_index + 1, time_step)
        if return_all:
            # return all frames between context_frame_idx and context_frame_idx + num_max_future_frames
            target_frame_idxs = np.arange(start_index, end_index + 1)
        else:
            # randomly sample "num_target_timesteps" frames
            target_frame_idxs = np.random.choice(
                np.arange(
                    context_frame_idxs[0],
                    context_frame_idxs[0] + int(fps),
                ),
                self.num_target_timesteps,
                replace=False,
            )
        target_frame_idxs = sorted(target_frame_idxs)
        return context_frame_idxs, np.array(target_frame_idxs)


class PerceptualModelDatasetEval(PerceptualModelDataset):
    def __init__(
        self,
        scene_id_list: Optional[List[int]] = None,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)

        if scene_id_list is None:
            scene_id_list = list(range(len(self.annotations)))
        val_sample_list = []
        for scene_id in scene_id_list:
            interval = self.get_interval(self.annotations[scene_id]['fps'])
            for start_id in range(0, self.annotations[scene_id]['num_timesteps'], interval):
                if scene_id == 63 and start_id == 0:
                    continue
                # Handle the last segment in the scene that does not meet the interval
                # (this has already been inferred in the previous few frames).
                if start_id + interval > self.annotations[scene_id]['num_timesteps']:
                    start_id = self.annotations[scene_id]['num_timesteps'] - interval
                val_sample_list.append((scene_id, start_id))
        self.val_sample_list = val_sample_list

    def __len__(self) -> int:
        return len(self.val_sample_list)

    def __getitem__(self, index: int):
        return super(PerceptualModelDatasetEval, self).__getitem__(
            self.val_sample_list[index][0],
            self.val_sample_list[index][1],
            return_all=True,
        )


class SingleSequenceDataset(PerceptualModelDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        scene_json = self.annotations[index % len(self.annotations)]
        num_timesteps = scene_json["num_timesteps"]
        if end_index < 0 or end_index > num_timesteps:
            end_index = num_timesteps
        segment_data = []
        cam_to_world = scene_json["camera_to_world"]
        ref_camera_name = DATASET_DICT[scene_json["dataset"]]["ref_camera"]

        # world_to_canonical
        world_to_canonical = np.linalg.inv(
            torch.tensor(cam_to_world[ref_camera_name][start_index])
        )  # dtype('float32')

        clip_start_id = None
        interval = self.get_interval(scene_json['fps'])
        for start_id in trange(start_index, end_index, interval):
            # Handle the last segment in the scene that does not meet the interval
            # (this has already been inferred in the previous few frames).
            if start_id + interval > num_timesteps:
                clip_start_id = start_id
                start_id = num_timesteps - interval
                logger.info(
                    "Total num_timesteps is %s, the last segment start at %s, will clip at %s.",
                    num_timesteps, start_id, clip_start_id
                )

            segment_dict = self.get_segment(
                index=index, context_frame_idx=start_id, return_all=True
            )
            # compute the relative transformation from a start_id to the first frame
            current_world = cam_to_world[ref_camera_name][start_id]

            segment_to_ref = (
                DATASETS[scene_json['dataset']]["canonical_to_flu"] @ world_to_canonical
            ) @ (
                current_world @ np.linalg.inv(
                    DATASETS[scene_json['dataset']]["canonical_to_flu"]
                )
            )

            segment_dict["segment_to_ref"] = to_float_tensor(segment_to_ref)
            if clip_start_id is not None:
                segment_dict["relative_clip_start_id"] = clip_start_id - start_id
            segment_data.append(segment_dict)
        return segment_data
