# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

This script is adapted for **Ascend OM-only** inference:
- Loads the OM model via AclLite.
- Builds OM inputs strictly from `config.json` input_features.
- Applies `policy_preprocessor.json` (observation normalization) and
    `policy_postprocessor.json` (action unnormalization) when available.
- Supports ACT-style chunked outputs (B, T, A) by replaying the chunk step-by-step.

"""

from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor
from tqdm import trange

# Ascend/ACL runtime dependencies (available on the target Ascend machine).
# We keep them optional so developers can still open/read this file locally.
_ASCEND_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover
    import acl  # type: ignore
    import acllite_utils as utils  # type: ignore
    import constants as const  # type: ignore
    from acllite_imageproc import AclLiteImageProc  # type: ignore
    from acllite_model import AclLiteModel  # type: ignore
    from acllite_resource import resource_list  # type: ignore
except Exception as exc:  # pragma: no cover
    _ASCEND_IMPORT_ERROR = exc
    acl = None  # type: ignore

    class _DummyUtils:  # pragma: no cover
        @staticmethod
        def display_time(fn):
            return fn

        @staticmethod
        def check_ret(*_args, **_kwargs):
            return None

    utils = _DummyUtils()  # type: ignore
    const = SimpleNamespace(SUCCESS=0)  # type: ignore
    AclLiteImageProc = None  # type: ignore
    AclLiteModel = None  # type: ignore
    resource_list = None  # type: ignore

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, check_env_attributes_and_types, close_envs, preprocess_observation
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)

from policy_input_schema import PolicyInputSchema, extract_policy_input_schema, load_config_json

try:  # Optional: only needed for nicer CLI error messages
    from draccus.utils import DecodingError as _DraccusDecodingError  # type: ignore
except Exception:  # pragma: no cover
    _DraccusDecodingError = None


_CLI_OM_MODEL_PATH: str | None = None


def _as_bool(x: object) -> bool | None:
    """Best-effort conversion of scalar-ish values to bool."""
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray) and x.shape == ():
        return bool(x.item())
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return bool(x.item())
    # numeric
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    return None


def _find_success_in_mapping(obj: object, *, max_depth: int = 4) -> bool | None:
    """Recursively search nested mappings for common success keys."""
    if max_depth <= 0:
        return None
    if not isinstance(obj, Mapping):
        return None

    for key in ("is_success", "success"):
        if key in obj:
            b = _as_bool(obj.get(key))
            if b is not None:
                return b

    for v in obj.values():
        if isinstance(v, Mapping):
            found = _find_success_in_mapping(v, max_depth=max_depth - 1)
            if found is not None:
                return found

    return None


def _extract_successes(
    *,
    info: Mapping[str, object],
    terminated: np.ndarray,
    truncated: np.ndarray,
    num_envs: int,
) -> list[bool]:
    """Extract per-env success flags for this step.

    Different env stacks report success differently:
    - VectorEnv terminal info: info['final_info'] (list of dicts)
    - Per-step vectors: info['success'] or info['is_success'] (len=num_envs)
    - Nested dicts under final_info entries

    We keep semantics: success is only counted on done steps.
    """
    done_step = np.asarray(terminated) | np.asarray(truncated)
    done_ix_global = np.where(done_step.reshape(-1).astype(bool))[0]

    # Some env stacks report success only inside `final_info` on terminal steps, while still exposing
    # a per-step `is_success` vector that can be all-False. To avoid false negatives, we parse both
    # and combine them (logical OR) for done envs.
    per_step_successes: list[bool] | None = None

    def _combine_final(final_successes: list[bool]) -> list[bool]:
        if per_step_successes is None:
            return final_successes
        out = [False] * num_envs
        for i in range(num_envs):
            out[i] = bool((final_successes[i] or per_step_successes[i]) and bool(done_step[i]))
        return out

    def _flat_values(x: object) -> np.ndarray | None:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            try:
                return x.detach().cpu().reshape(-1).numpy()
            except Exception:
                return None
        if isinstance(x, np.ndarray):
            return x.reshape(-1)
        if isinstance(x, (list, tuple)):
            return np.asarray(list(x)).reshape(-1)
        # scalar
        try:
            return np.asarray([x]).reshape(-1)
        except Exception:
            return None

    def _done_env_indices() -> np.ndarray:
        # Gymnasium VectorEnv often provides a mask under _final_info.
        m = info.get("_final_info")
        if isinstance(m, np.ndarray):
            mm = np.asarray(m).astype(bool).reshape(-1)
            if mm.shape[0] == num_envs:
                return np.where(mm)[0]
        return done_ix_global

    def _assign_with_mask(values: object, mask: object) -> list[bool] | None:
        """Assign values to env indices specified by a boolean mask.

        In Gymnasium VectorEnv, a key `k` can be provided only for done envs as:
          info[k] = values_for_done (len = num_done)
          info[_k] = done_mask (len = num_envs)
        """
        vv = _flat_values(values)
        mm = _flat_values(mask)
        if vv is None or mm is None:
            return None
        try:
            mm_bool = np.asarray(mm).astype(bool).reshape(-1)
        except Exception:
            return None
        out = [False] * num_envs

        # Case 1: mask is over all envs (len=num_envs), values for selected envs.
        if mm_bool.shape[0] == num_envs:
            # Variant A: values are also per-env (len=num_envs). In this case the mask indicates which
            # entries are valid; we should not require values to be compacted to only the True entries.
            if vv.shape[0] == num_envs:
                for env_i in range(num_envs):
                    if not bool(mm_bool[env_i]):
                        continue
                    b = _as_bool(vv[env_i])
                    out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
                return out

            # Variant B: values are compacted to only the True entries.
            idx = np.where(mm_bool)[0]
            if vv.shape[0] != idx.shape[0]:
                return None
            for j, env_i in enumerate(idx.tolist()):
                b = _as_bool(vv[j])
                out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
            return out

        # Case 2: mask is over done envs only (len=num_done), values for selected done envs.
        done_ix = done_ix_global
        if mm_bool.shape[0] == done_ix.shape[0]:
            local_idx = np.where(mm_bool)[0]
            # values might be provided for all done envs, or only for selected (True) entries.
            if vv.shape[0] == done_ix.shape[0]:
                vv_sel = vv[local_idx]
            elif vv.shape[0] == local_idx.shape[0]:
                vv_sel = vv
            else:
                return None
            for j, local_j in enumerate(local_idx.tolist()):
                env_i = int(done_ix[local_j])
                b = _as_bool(vv_sel[j])
                out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
            return out

        return None

    def _assign_direct(values: object) -> list[bool] | None:
        vv = _flat_values(values)
        if vv is None:
            return None
        if vv.shape[0] != num_envs:
            return None
        out = [False] * num_envs
        for i in range(num_envs):
            b = _as_bool(vv[i])
            out[i] = bool((b if b is not None else False) and bool(done_step[i]))
        return out

    # 1) Per-step success keys, supporting Gymnasium mask format.
    for key in ("is_success", "success"):
        if key in info:
            masked = _assign_with_mask(info.get(key), info.get(f"_{key}"))
            if masked is not None:
                per_step_successes = masked
                break
            direct = _assign_direct(info.get(key))
            if direct is not None:
                per_step_successes = direct
                break

    # 2) VectorEnv terminal info.
    final_info = info.get("final_info")
    if isinstance(final_info, np.ndarray):
        # gymnasium sometimes returns object arrays here.
        try:
            final_info = list(final_info)
        except Exception:
            final_info = None

    # 2a) Common: final_info is list/tuple (either len=num_envs or len=num_done).
    if isinstance(final_info, (list, tuple)):
        done_ix = _done_env_indices()
        if len(final_info) == num_envs:
            out: list[bool] = []
            for i, entry in enumerate(final_info):
                if entry is None or not bool(done_step[i]):
                    out.append(False)
                    continue
                if isinstance(entry, Mapping):
                    found = _find_success_in_mapping(entry)
                    out.append(bool(found) if found is not None else False)
                else:
                    out.append(False)
            return _combine_final(out)
        if len(final_info) == len(done_ix):
            out = [False] * num_envs
            for j, env_i in enumerate(done_ix.tolist()):
                entry = final_info[j]
                if entry is None or not bool(done_step[env_i]):
                    continue
                if isinstance(entry, Mapping):
                    found = _find_success_in_mapping(entry)
                    out[env_i] = bool(found) if found is not None else False
            return _combine_final(out)

    # 2b) Sometimes: final_info is a dict-of-arrays (stacked info for done envs), possibly with nested masks.
    if isinstance(final_info, Mapping):
        done_ix = _done_env_indices()
        # Look for success keys inside final_info.
        for key in ("is_success", "success"):
            if key not in final_info:
                continue
            vv = _flat_values(final_info.get(key))
            if vv is None:
                continue

            inner_mask = final_info.get(f"_{key}")
            if inner_mask is not None:
                mm = _flat_values(inner_mask)
                if mm is not None:
                    mm_bool = np.asarray(mm).astype(bool).reshape(-1)
                    if mm_bool.shape[0] == done_ix.shape[0] and vv.shape[0] == done_ix.shape[0]:
                        done_ix2 = done_ix[mm_bool]
                        vv2 = vv[mm_bool]
                    else:
                        done_ix2 = done_ix
                        vv2 = vv
                else:
                    done_ix2 = done_ix
                    vv2 = vv
            else:
                done_ix2 = done_ix
                vv2 = vv

            if vv2.shape[0] != done_ix2.shape[0]:
                continue

            out = [False] * num_envs
            for j, env_i in enumerate(done_ix2.tolist()):
                b = _as_bool(vv2[j])
                out[env_i] = bool((b if b is not None else False) and bool(done_step[env_i]))
            return _combine_final(out)

        # As a last resort, try recursive search if final_info looks like a single env dict.
        found = _find_success_in_mapping(final_info)
        if found is not None and bool(done_step.any()):
            # If we can't map per-env, at least mark done envs as this value.
            out = [False] * num_envs
            for env_i in _done_env_indices().tolist():
                out[env_i] = bool(found) and bool(done_step[env_i])
            return _combine_final(out)

    # If we got per-step successes but couldn't extract any final_info, return per-step.
    if per_step_successes is not None:
        return per_step_successes

    # 3) Fallback.
    return [False] * num_envs


def _strip_cli_arg(argv: list[str], *, key: str) -> tuple[list[str], str | None]:
    """Remove an argument from argv and return (new_argv, value).

    Supports both `--key value` and `--key=value` forms.
    """
    out: list[str] = []
    value: str | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == key:
            if i + 1 >= len(argv):
                raise ValueError(f"{key} requires a value")
            value = argv[i + 1]
            i += 2
            continue
        if tok.startswith(key + "="):
            value = tok.split("=", 1)[1]
            i += 1
            continue
        out.append(tok)
        i += 1
    return out, value


def _preparse_om_model_path_from_argv() -> None:
    """Capture --om_model_path without requiring EvalPipelineConfig to declare it.

    This is needed because some run environments may have an older `lerobot` installed where
    EvalPipelineConfig doesn't include `om_model_path`, causing Draccus to error on unknown args.
    """
    global _CLI_OM_MODEL_PATH
    argv = list(sys.argv)

    for key in ("--om_model_path", "--om-model-path"):
        argv, val = _strip_cli_arg(argv, key=key)
        if val is not None:
            _CLI_OM_MODEL_PATH = val
            break

    # IMPORTANT: persist the stripped argv so Draccus doesn't see unknown args.
    sys.argv = argv


def _unwrap_single_vec_env(envs: object) -> gym.vector.VectorEnv:
    """Unwrap `make_env()` outputs to a single VectorEnv.

    In current LeRobot, `make_env()` returns `{suite: {task_id: vec_env}}`.
    This evaluator only supports a single suite + single task.
    """

    if isinstance(envs, gym.vector.VectorEnv):
        return envs

    # Most common: {suite: {task_id: vec_env}}
    if isinstance(envs, Mapping):
        if len(envs) != 1:
            raise ValueError(
                "This evaluator expects exactly 1 env suite, but got suites: "
                f"{list(envs.keys())}."
            )
        suite_envs = next(iter(envs.values()))
        if isinstance(suite_envs, gym.vector.VectorEnv):
            return suite_envs
        if isinstance(suite_envs, Mapping):
            if len(suite_envs) != 1:
                raise ValueError(
                    "This evaluator expects exactly 1 task env, but got task_ids: "
                    f"{list(suite_envs.keys())}."
                )
            vec = next(iter(suite_envs.values()))
            if isinstance(vec, gym.vector.VectorEnv):
                return vec
            raise TypeError(f"Unexpected task env type: {type(vec)}")
        raise TypeError(f"Unexpected suite env container type: {type(suite_envs)}")

    # Occasionally: [env] / (env, ...)
    if isinstance(envs, Sequence) and not isinstance(envs, (str, bytes)):
        if len(envs) != 1:
            raise ValueError(f"Expected a single env, but got {len(envs)} items")
        return _unwrap_single_vec_env(envs[0])

    raise TypeError(f"Unsupported make_env() return type: {type(envs)}")


class AclLiteResource:
    """
    AclLiteResource
    """
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self):
        """
        init resource
        """
        ret = acl.init()

        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

    def __del__(self):
        resource_list.destroy()
        if self.stream:
            acl.rt.destroy_stream(self.stream)

        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)


class ACT(object):
    """
    class for ACT
    """
    def __init__(self, model_path, model_width, model_height):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self.device_id = 0
        self._dvpp = None
        self._model = None
        # For ACT-style models that output an action chunk (B, T, A), we can
        # execute the OM model once and then play back the chunk step-by-step.
        self._chunk_actions: np.ndarray | None = None  # (T, B, A)
        self._chunk_pos: int = 0

    def init(self):
        """
        Initialize
        """
        # 初始化dvpp
        self._dvpp = AclLiteImageProc()

        # 加载模型
        self._model = AclLiteModel(self._model_path)
        return const.SUCCESS

    def inference(self, input_list):
        """
        model inference
        """
        if not isinstance(input_list, list) or not input_list:
            raise ValueError("input_list must be a non-empty list of numpy arrays")

        batch_size = int(getattr(input_list[0], "shape", [0])[0] or 0)
        if batch_size <= 0:
            raise ValueError(f"Invalid batch size inferred from input_list[0]: {getattr(input_list[0], 'shape', None)}")

        # ACT policies often output an action chunk (B, T, A). If we always take only the
        # first step, actions can become nearly constant ("moves once then holds").
        # Default behavior: if output is chunked, play back the chunk step-by-step.
        use_chunk_queue = _env_flag_bool("LEROBOT_ACT_USE_CHUNK_QUEUE", "1")

        if use_chunk_queue and self._chunk_actions is not None:
            if self._chunk_pos < self._chunk_actions.shape[0]:
                action = self._chunk_actions[self._chunk_pos]
                self._chunk_pos += 1
                return np.ascontiguousarray(action)
            # chunk exhausted
            self._chunk_actions = None

        # Execute the model (either non-chunked output, or chunk needs refresh)
        outputs = self._model.execute(input_list)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 1:
            raise RuntimeError(f"Unexpected OM outputs: {type(outputs)}")
        out = np.asarray(outputs[0])

        if out.ndim == 3:
            # (B, T, A)
            if out.shape[0] != batch_size:
                raise ValueError(f"OM output batch mismatch: got {out.shape[0]} expected {batch_size}")
            if use_chunk_queue:
                # Store as (T, B, A) for easy indexing per step.
                self._chunk_actions = np.ascontiguousarray(np.transpose(out, (1, 0, 2)))
                self._chunk_pos = 1
                action = self._chunk_actions[0]
            else:
                # Fallback: take the first step only.
                action = out[:, 0, :]
        elif out.ndim == 2:
            # (B, action_dim)
            if out.shape[0] != batch_size:
                raise ValueError(f"OM output batch mismatch: got {out.shape[0]} expected {batch_size}")
            action = out
        elif out.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"OM output is 1D but batch_size={batch_size}")
            action = out[None, :]
        else:
            raise ValueError(f"Unsupported OM output shape: {out.shape}")
        return np.ascontiguousarray(action)


def act_init(model_path, model_width, model_height):
    """
    init act resource and act model
    """
    act = ACT(model_path, model_width, model_height)
    ret = act.init()
    utils.check_ret("ACT init", ret)
    return act


def _resolve_om_model_path(om_model_path: Path | None, *, policy_dir: Path) -> Path:
    if om_model_path is None:
        raise ValueError("Missing --om_model_path (Path to OM model)")

    p = Path(om_model_path).expanduser()
    resolved = (policy_dir / p).resolve() if not p.is_absolute() else p.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"OM model path does not exist: {resolved}")
    return resolved


def _first_visual_hw(schema: PolicyInputSchema) -> tuple[int, int]:
    if not schema.visual_keys:
        raise ValueError("No VISUAL inputs in schema")
    shape = schema.input_shapes[schema.visual_keys[0]]
    if len(shape) != 3:
        raise ValueError(f"Unexpected VISUAL shape for {schema.visual_keys[0]}: {shape}")
    _, h, w = shape
    return int(h), int(w)


def _find_policy_postprocessor_dir(policy_dir: Path) -> Path | None:
    """Locate a directory containing policy_postprocessor.json (+ state safetensors).

    We prefer:
      1) policy_dir/ (already migrated/included)
      2) sibling policy_dir_parent/<policy_dir_name>_migrated/
    """
    direct = policy_dir / "policy_postprocessor.json"
    if direct.exists():
        return policy_dir

    migrated = policy_dir.parent / f"{policy_dir.name}_migrated" / "policy_postprocessor.json"
    if migrated.exists():
        return migrated.parent

    return None


def _find_policy_preprocessor_dir(policy_dir: Path) -> Path | None:
    """Locate a directory containing policy_preprocessor.json (+ state safetensors).

    We prefer:
      1) policy_dir/ (already migrated/included)
      2) sibling policy_dir_parent/<policy_dir_name>_migrated/
    """
    direct = policy_dir / "policy_preprocessor.json"
    if direct.exists():
        return policy_dir

    migrated = policy_dir.parent / f"{policy_dir.name}_migrated" / "policy_preprocessor.json"
    if migrated.exists():
        return migrated.parent

    return None


def _maybe_load_policy_preprocessor(policy_dir: Path) -> PolicyProcessorPipeline | None:
    pre_dir = _find_policy_preprocessor_dir(policy_dir)
    if pre_dir is None:
        return None
    try:
        return PolicyProcessorPipeline.from_pretrained(pre_dir, config_filename="policy_preprocessor.json")
    except Exception as err:
        logging.warning(
            "Failed to load policy_preprocessor.json from %s. "
            "This usually means the referenced state *.safetensors file(s) are missing or incompatible. "
            "Error: %s",
            pre_dir,
            err,
        )
        return None


def _maybe_load_policy_postprocessor(policy_dir: Path) -> PolicyProcessorPipeline | None:
    post_dir = _find_policy_postprocessor_dir(policy_dir)
    if post_dir is None:
        return None
    try:
        return PolicyProcessorPipeline.from_pretrained(post_dir, config_filename="policy_postprocessor.json")
    except Exception as err:
        logging.warning(
            "Failed to load policy_postprocessor.json from %s. "
            "This usually means the referenced state *.safetensors file(s) are missing or incompatible. "
            "Error: %s",
            post_dir,
            err,
        )
        return None


def _load_policy_processor_or_raise(
    *,
    policy_dir: Path,
    kind: str,
    filename: str,
) -> PolicyProcessorPipeline:
    """Load a processor pipeline and raise a useful error if it cannot be loaded.

    We keep this evaluator strict (hard fail) because silent fallbacks lead to wrong actions.
    """
    if kind == "pre":
        proc_dir = _find_policy_preprocessor_dir(policy_dir)
    elif kind == "post":
        proc_dir = _find_policy_postprocessor_dir(policy_dir)
    else:
        raise ValueError(f"Unknown processor kind: {kind}")

    expected_a = policy_dir / filename
    expected_b = policy_dir.parent / f"{policy_dir.name}_migrated" / filename

    if proc_dir is None:
        raise FileNotFoundError(
            f"Missing {filename}. Expected one of:\n"
            f"- {expected_a}\n"
            f"- {expected_b}\n\n"
            "Generate it with the processor migration script (creates *.json + *.safetensors), "
            "then re-run evaluation."
        )

    try:
        return PolicyProcessorPipeline.from_pretrained(proc_dir, config_filename=filename)
    except Exception as err:
        raise RuntimeError(
            f"Found {filename} under {proc_dir}, but failed to load it. "
            "Most commonly, the referenced state *.safetensors file(s) were not copied alongside the json, "
            "or the files are incompatible with the installed lerobot version. "
            f"Original error: {err}"
        ) from err


def _env_flag(name: str, default: str = "0") -> str:
    v = str(os.environ.get(name, default)).strip()
    return v


def _env_flag_bool(name: str, default: str = "0") -> bool:
    v = _env_flag(name, default)
    return v not in ("0", "false", "False", "no", "NO", "")


def _to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype, copy=False)
    return np.ascontiguousarray(x)


@dataclass
class RolloutParams:
    env: gym.vector.VectorEnv
    policy: PreTrainedPolicy | None
    seeds: list[int] | None = None
    return_observations: bool = False
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None
    model: ACT | None = None
    om_schema: PolicyInputSchema | None = None
    policy_preprocessor: PolicyProcessorPipeline | None = None
    policy_postprocessor: PolicyProcessorPipeline | None = None


@dataclass
class EvalPolicyParams:
    env: gym.vector.VectorEnv
    policy: PreTrainedPolicy | None
    n_episodes: int
    max_episodes_rendered: int = 0
    videos_dir: Path | None = None
    return_episode_data: bool = False
    start_seed: int | None = None
    model: ACT | None = None
    om_schema: PolicyInputSchema | None = None
    policy_preprocessor: PolicyProcessorPipeline | None = None
    policy_postprocessor: PolicyProcessorPipeline | None = None


def build_om_input_list(obs: dict[str, torch.Tensor]) -> list[np.ndarray]:
    """
    将预处理后的 observation 字典打包为 Ascend OM 模型所需的输入列表。

    输入顺序严格按 policy config.json 的 input_features 顺序(schema.input_keys)。
    """
    raise RuntimeError("build_om_input_list(obs) is deprecated; use build_om_input_list_from_schema(obs, schema).")


def build_om_input_list_from_schema(obs: dict[str, torch.Tensor], schema: PolicyInputSchema) -> list[np.ndarray]:
    inputs: list[np.ndarray] = []
    for key in schema.input_keys:
        if key not in obs:
            raise KeyError(f"Missing observation key '{key}' required by config.json input_features")
        inputs.append(_to_numpy(obs[key], dtype=np.float32))
    return inputs


def rollout(params: RolloutParams) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        params: Rollout configuration and dependencies.
    Returns:
        The dictionary described above.
    """
    env = params.env
    policy = params.policy
    seeds = params.seeds
    return_observations = params.return_observations
    render_callback = params.render_callback
    model = params.model
    om_schema = params.om_schema
    policy_preprocessor = params.policy_preprocessor
    policy_postprocessor = params.policy_postprocessor
    device = "cpu"
    # Reset the policy and environments.
    # policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        observation = {
            key: observation[key].to(device, non_blocking=False) for key in observation
        }

        # Infer "task" from attributes of environments.
        # Note: add_envs_task currently supports SyncVectorEnv; extend if AsyncVectorEnv needed.
        observation = add_envs_task(env, observation)

        # Apply the training-time policy preprocessor (e.g., mean/std normalization).
        # This is critical for ACT models trained with external processor pipelines.
        if policy_preprocessor is not None:
            observation = policy_preprocessor.process_observation(observation)

        # with torch.inference_mode():
        #     action = policy.select_action(observation)
            # change the logic here if the device is npu
        if model is None or om_schema is None:
            raise ValueError("Ascend OM rollout requires model and om_schema")

        input_list = build_om_input_list_from_schema(observation, om_schema)
        action = model.inference(input_list)

        if policy_postprocessor is not None:
            # Postprocessor operates on an EnvTransition; use the pipeline convenience method.
            action_t = torch.from_numpy(np.asarray(action)).to(dtype=torch.float32)
            action_t = policy_postprocessor.process_action(action_t)
            if not isinstance(action_t, torch.Tensor):
                raise TypeError(f"Postprocessor returned unexpected type: {type(action_t)}")
            action = np.ascontiguousarray(action_t.detach().cpu().numpy())

        # Convert to CPU / numpy.
        # action = action.to("cpu").numpy()
        if action.ndim != 2:
            raise ValueError(f"Action dimensions should be (batch, action_dim), got {action.shape}")
        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)

        # Success reporting differs across env stacks (final_info vs per-step vectors).
        # Parse defensively so success_rate doesn't get stuck at 0 when episodes succeed.
        successes = _extract_successes(
            info=info,
            terminated=terminated,
            truncated=truncated,
            num_envs=env.num_envs,
        )

        # Keep track of which environments are done so far.
        done = terminated | truncated | done

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    if policy is not None and hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(params: EvalPolicyParams) -> dict:
    """
    Args:
        params: Evaluation configuration and dependencies.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    env = params.env
    policy = params.policy
    n_episodes = params.n_episodes
    max_episodes_rendered = params.max_episodes_rendered
    videos_dir = params.videos_dir
    return_episode_data = params.return_episode_data
    start_seed = params.start_seed
    model = params.model
    om_schema = params.om_schema
    policy_preprocessor = params.policy_preprocessor
    policy_postprocessor = params.policy_postprocessor
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    # if not isinstance(policy, PreTrainedPolicy):
    #     raise ValueError(
    #         f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
    #     )

    start = time.time()
    # policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_params = RolloutParams(
            env=env,
            policy=policy,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            model=model,
            om_schema=om_schema,
            policy_preprocessor=policy_preprocessor,
            policy_postprocessor=policy_postprocessor,
        )
        rollout_data = rollout(rollout_params)

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                if episode_data["episode_index"][-1] + 1 != this_episode_data["episode_index"][0]:
                    raise ValueError("Episode indices are not consecutive when compiling episode data.")
                if episode_data["index"][-1] + 1 != this_episode_data["index"][0]:
                    raise ValueError("Data indices are not consecutive when compiling episode data.")
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    per_episode: list[dict[str, object]] = []
    for idx, (sum_reward, max_reward, success, seed) in enumerate(
        zip(
            sum_rewards[:n_episodes],
            max_rewards[:n_episodes],
            all_successes[:n_episodes],
            all_seeds[:n_episodes],
            strict=True,
        )
    ):
        per_episode.append(
            {
                "episode_ix": idx,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
        )

    # Compile eval info.
    info = {
        "per_episode": per_episode,
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data["action"].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            "action": rollout_data["action"][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            "next.done": rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            "next.reward": rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data["observation"]:
            ep_dict[key] = rollout_data["observation"][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    if acl is None or AclLiteModel is None or AclLiteImageProc is None:
        raise RuntimeError(
            "Ascend ACL runtime dependencies are not available in this Python environment. "
            "Run this script on the Ascend target machine with the ACL/AclLite Python packages installed. "
            f"Original import error: {_ASCEND_IMPORT_ERROR}"
        )

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    env = _unwrap_single_vec_env(envs)

    # OM-only evaluation: do NOT build/load the PyTorch policy (which would require model.safetensors).
    # We only need cfg.policy for device/amp flags and for locating config.json (input schema).
    policy = None
    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError("cfg.policy.pretrained_path is required to load config.json for OM inputs")

    policy_dir = Path(cfg.policy.pretrained_path).expanduser().resolve()
    policy_config_path = policy_dir / "config.json"
    policy_config = load_config_json(policy_config_path)
    om_schema = extract_policy_input_schema(policy_config)

    policy_preprocessor = _load_policy_processor_or_raise(
        policy_dir=policy_dir,
        kind="pre",
        filename="policy_preprocessor.json",
    )
    logging.info("Loaded policy_preprocessor.json (will preprocess OM inputs)")

    policy_postprocessor = _load_policy_processor_or_raise(
        policy_dir=policy_dir,
        kind="post",
        filename="policy_postprocessor.json",
    )
    logging.info("Loaded policy_postprocessor.json (will unnormalize OM actions)")

    cfg_om_path = getattr(cfg, "om_model_path", None)
    if cfg_om_path is None and _CLI_OM_MODEL_PATH is not None:
        cfg_om_path = Path(_CLI_OM_MODEL_PATH)
    model_path = _resolve_om_model_path(cfg_om_path, policy_dir=policy_dir)
    h, w = _first_visual_hw(om_schema)
    logging.info(f"Loaded config.json: {policy_config_path}")
    logging.info(f"OM model path: {model_path}")
    logging.info(f"OM expects VISUAL HxW: {h}x{w}; inputs: {list(om_schema.input_keys)}")

    acl_resource = AclLiteResource()
    acl_resource.init()

    model = act_init(str(model_path), w, h)
    eval_params = EvalPolicyParams(
        env=env,
        policy=policy,
        n_episodes=cfg.eval.n_episodes,
        max_episodes_rendered=10,
        videos_dir=Path(cfg.output_dir) / "videos",
        start_seed=cfg.seed,
        model=model,
        om_schema=om_schema,
        policy_preprocessor=policy_preprocessor,
        policy_postprocessor=policy_postprocessor,
    )
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy(eval_params)
    logging.info("Aggregated eval metrics: %s", info.get("aggregated"))

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    close_envs(envs)

    logging.info("End of eval")


def main():
    init_logging()
    _preparse_om_model_path_from_argv()
    try:
        eval_main()
    except Exception as err:
        # Draccus raises before calling eval_main(cfg) if required fields are missing.
        if _DraccusDecodingError is not None and isinstance(err, _DraccusDecodingError):
            logging.error(
                "Missing required evaluation config. You must provide at least `env` and `--om_model_path`. "
                "Use either an eval config file via `--config=<path>` or provide env fields on CLI "
                "(e.g. `--env.type=...`).\n"
                "Example:\n"
                "  python3 %s --policy.path=/path/to/pretrained_model "
                "--om_model_path=/path/to/act_fp16.om --env.type=<ENV>",
                Path(sys.argv[0]).name,
            )
            raise SystemExit(2) from err

        # Our explicit runtime checks.
        msg = str(err)
        if "Missing --om_model_path" in msg:
            logging.error(
                "Missing required OM model path. Provide `--om_model_path=/path/to/model.om` "
                "(relative paths are resolved under `--policy.path`)."
            )
            raise SystemExit(2) from err
        if "cfg.policy.pretrained_path is required" in msg:
            logging.error(
                "Missing policy path. Provide `--policy.path=/path/to/pretrained_model` "
                "(must contain config.json)."
            )
            raise SystemExit(2) from err

        raise


if __name__ == "__main__":
    main()
