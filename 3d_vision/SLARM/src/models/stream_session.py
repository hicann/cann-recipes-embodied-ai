# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from src.models.slarm import SLARM

# Keys whose predictions are accumulated (concatenated) across stream steps.
_ACCUMULATED_PREDICTION_KEYS = [
    'gs_params', 'pred_feat', 'sky_token', 'affine_tokens',
    'pred_context_depth', 'pred_context_camera_enc_list',
    'pred_context_depth_conf', 'pred_context_pts3d',
    'pred_context_pts3d_conf',
]

# Keys that should be fully replaced (not concatenated) on each step.
_REPLACE_KEYS = {'sky_token', 'affine_tokens'}


class StreamSession:
    """
    A causal streaming inference session with KV cache management.
    """

    def __init__(self, model: SLARM, mode: str, window_size=4):
        self.model = model
        self.mode = mode
        self.window_size = window_size

        self.aggregator_kv_cache_depth = self.model.aggregator.depth
        self.camera_head_kv_cache_depth = (
            self.model.camera_head.trunk_depth
            if self.model.camera_head is not None
            else 0
        )
        self.camera_head_iterations = (
            4 if self.model.camera_head is not None else 0
        )

        if self.mode not in ["causal", "window"]:
            raise ValueError(
                f"Unsupported attention mode when using kv_cache: "
                f"{self.mode}"
            )

        # Instance attributes initialised at construction (G.CLS.08)
        self.predictions: dict = {}
        self.aggregator_kv_cache_list: List = []
        self.camera_head_kv_cache_list: Optional[List] = None

        self.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_predictions(self):
        return self.predictions

    def get_last_prediction(self):
        last_predictions = {}
        for k in [
            "pose_enc", "world_points", "world_points_conf",
            "depth", "depth_conf", "images",
        ]:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][:, -1:]
        return last_predictions

    def clear(self):
        self._clear_predictions()
        self._clear_cache()

    def forward_stream(self, input_dict, device, dtype):
        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = self.model(
                    input_dict,
                    stream_save=False,
                    aggregator_kv_cache_list=aggregator_kv_cache_list,
                    camera_head_kv_cache_list=camera_head_kv_cache_list,
                )

        self._update_predictions(outputs)
        required_keys = [
            "aggregator_kv_cache_list", "camera_head_kv_cache_list"
        ]
        for key in required_keys:
            if key not in outputs:
                raise ValueError(f"{key} not found in model outputs")
        self._update_cache(
            outputs["aggregator_kv_cache_list"],
            outputs["camera_head_kv_cache_list"],
        )

        return self.get_all_predictions()

    # ------------------------------------------------------------------
    # Prediction management
    # ------------------------------------------------------------------

    def _clear_predictions(self):
        self.predictions = {k: None for k in _ACCUMULATED_PREDICTION_KEYS}

    def _update_predictions(self, predictions):
        for k in _ACCUMULATED_PREDICTION_KEYS:
            if k not in predictions:
                continue
            pred_value = predictions[k]
            self._accumulate_prediction(k, pred_value)

    def _accumulate_prediction(self, key, pred_value):
        """Merge a new prediction value for *key* into self.predictions."""
        # First-time assignment
        if self.predictions.get(key, None) is None:
            self.predictions[key] = pred_value
            return

        # Keys that are replaced entirely each step
        if key in _REPLACE_KEYS:
            self.predictions[key] = pred_value
            return

        # Camera enc list: concat per-iteration item
        if key == 'pred_context_camera_enc_list':
            for i, item in enumerate(self.predictions[key]):
                self.predictions[key][i] = torch.cat(
                    [item, pred_value[i]], dim=1
                )
            return

        if key == 'gs_params':
            for param_key in pred_value.keys():
                if param_key == 'motion_bases':
                    self.predictions['gs_params']['motion_bases'] = (
                        pred_value['motion_bases']
                    )
                elif param_key == 'affine':
                    continue
                else:
                    current = self.predictions['gs_params'].get(
                        param_key, None
                    )
                    self.predictions['gs_params'][param_key] = torch.cat(
                        [current, pred_value[param_key]], dim=1
                    )
            return

        current = self.predictions.get(key, None)
        self.predictions[key] = torch.cat([current, pred_value], dim=1)

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [
            [None, None] for _ in range(self.aggregator_kv_cache_depth)
        ]
        self.camera_head_kv_cache_list = self._make_empty_camera_cache()

    def _make_empty_camera_cache(self):
        if self.model.camera_head is None:
            return None
        return [
            [
                [None, None] for _ in range(self.camera_head_kv_cache_depth)
            ]
            for _ in range(self.camera_head_iterations)
        ]

    def _update_cache(self, aggregator_kv_cache_list,
                      camera_head_kv_cache_list):
        if self.mode == "causal":
            self.aggregator_kv_cache_list = aggregator_kv_cache_list
            self.camera_head_kv_cache_list = camera_head_kv_cache_list
        elif self.mode == "window":
            self._update_cache_window(
                aggregator_kv_cache_list, camera_head_kv_cache_list
            )
        else:
            raise ValueError(
                f"Unsupported attention mode when using kv_cache: "
                f"{self.mode}"
            )

    def _update_cache_window(self, aggregator_kv_cache_list,
                             camera_head_kv_cache_list):
        """Slide the KV cache window: drop oldest frame, keep the rest."""
        window_size = self.window_size
        per_frame_lens = (
            aggregator_kv_cache_list[0][0].shape[2] // window_size
        )

        # Each kv_cache entry is [k_tensor, v_tensor]; index 0/1 = k/v
        for slot in range(2):
            self._slide_aggregator_cache(
                aggregator_kv_cache_list, slot, per_frame_lens
            )
            self._slide_camera_head_cache(
                camera_head_kv_cache_list, slot
            )

    def _slide_aggregator_cache(self, src_cache, slot, per_frame_lens):
        for i in range(self.aggregator_kv_cache_depth):
            self.aggregator_kv_cache_list[i][slot] = (
                src_cache[i][slot][:, :, per_frame_lens:]
            )

    def _slide_camera_head_cache(self, src_cache, slot):
        if self.camera_head_kv_cache_list is None:
            return
        for i in range(self.camera_head_iterations):
            for j in range(self.camera_head_kv_cache_depth):
                self.camera_head_kv_cache_list[i][j][slot] = (
                    src_cache[i][j][slot][:, :, 1:]
                )

    def _get_cache(self):
        return self.aggregator_kv_cache_list, self.camera_head_kv_cache_list
