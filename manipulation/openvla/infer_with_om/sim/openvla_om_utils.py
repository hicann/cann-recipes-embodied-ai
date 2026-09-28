# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (R) @huawei.com; all rights reserved
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

import acl
import acllite_utils as utils
import constants as const
from acllite_imageproc import AclLiteImageProc
from acllite_model import AclLiteModel
from acllite_kv_model import KVStatefulModel
from acllite_resource import resource_list

from vla_validation_utils import post_process_action


# ==================== Data Classes ====================

@dataclass
class ModelPaths:
    """Container for model file paths."""
    vision_backbone_path: str
    projector_path: str
    embedding_path: str
    prefill_model_path: str
    decode_model_path: str


@dataclass
class ActionConfig:
    """Container for action-related configuration."""
    action_dim: int
    vocab_size: int
    bin_centers: np.ndarray
    action_norm_stats: Dict


@dataclass
class ModelDimensions:
    """Container for model input dimensions."""
    width: int = 224
    height: int = 224


@dataclass
class OpenVLAConfig:
    """Container for OpenVLA model configuration."""
    model_paths: ModelPaths
    action_config: ActionConfig
    model_dims: ModelDimensions
    device_id: int = 0


class AclLiteResource:
    """
    ACL runtime resource manager.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self) -> None:
        """
        Init ACL resources.
        """
        logging.info("init resource stage:")
        _ = acl.init()

        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

        logging.info("Init resource success")

    def __del__(self):
        logging.info("acl resource release all resource")
        resource_list.destroy()
        if self.stream:
            logging.info("acl resource release stream")
            acl.rt.destroy_stream(self.stream)

        if self.context:
            logging.info("acl resource release context")
            acl.rt.destroy_context(self.context)

        logging.info("Reset acl device %s", self.device_id)
        acl.rt.reset_device(self.device_id)
        logging.info("Release acl resource success")


class VisionModel:
    """
    Wrapper for vision-related OM models (vision_backbone/projector/embedding).
    """

    def __init__(self, model_path: str, model_width: int, model_height: int):
        self.model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self.device_id = 0
        self._dvpp = None
        self._model = None

    def init(self) -> int:
        """
        Initialize DVPP and load OM model.
        """
        self._dvpp = AclLiteImageProc()
        self._model = AclLiteModel(self.model_path)
        return const.SUCCESS

    @utils.display_time
    def inference(self, input_list):
        """
        Run model inference.

        Args:
            input_list: List of input arrays

        Returns:
            First output array from the model
        """
        outputs = self._model.execute(input_list)
        return outputs[0]


class OpenVLA:
    """
    Wrapper for OpenVLA OM inference (prefill + decode) and post-process actions.
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        vision_backbone: VisionModel,
        projector: VisionModel,
        embedding: VisionModel,
        action_config: ActionConfig,
    ):
        self.model_paths = model_paths
        self.device_id = 0
        self._dvpp = None

        self.vision_backbone = vision_backbone
        self.projector = projector
        self.embedding = embedding

        self.prefill_model = None
        self.decode_model = None

        self.action_config = action_config
        self.action_dim = action_config.action_dim
        self.vocab_size = action_config.vocab_size
        self.bin_centers = action_config.bin_centers
        self.action_norm_stats = action_config.action_norm_stats

        self.max_new_tokens = action_config.action_dim  # keep original behavior

    def init(self) -> int:
        """
        Initialize DVPP and load OM models.
        """
        self._dvpp = AclLiteImageProc()
        self.prefill_model = KVStatefulModel(self.model_paths.prefill_model_path)
        self.decode_model = KVStatefulModel(self.model_paths.decode_model_path)
        return const.SUCCESS

    @utils.display_time
    def pre_process(self, input_list):
        """
        Pre-process inputs: ensure input_ids ends with pad token (29871).

        Args:
            input_list: Dictionary containing input_ids, pixel_values, attention_mask

        Returns:
            Tuple of (input_ids, pixel_values, attention_mask)
        """
        input_ids = input_list["input_ids"]
        pixel_values = input_list["pixel_values"]
        attention_mask = input_list["attention_mask"]

        if not torch.all(input_ids[:, -1] == 29871):
            pad_token_tensor = torch.tensor(
                [[29871]], dtype=torch.long, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, pad_token_tensor], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
        return input_ids, pixel_values, attention_mask

    @utils.display_time
    def inference(self, input_ids, pixel_values, attention_mask, target_seq_len: int = 288):
        """
        Perform model inference with all models (vision_backbone + projector + embedding + prefill + decode).

        Args:
            input_ids: Input token IDs
            pixel_values: Pixel values
            attention_mask: Attention mask
            target_seq_len: Target sequence length for padding
        """

        # Process vision through vision backbone and projector
        pixel_values = pixel_values.numpy()
        patch_features = self.vision_backbone.inference([pixel_values])
        projected_patch_embeddings = self.projector.inference([patch_features])
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # Get input embeddings from language model
        input_ids = input_ids.numpy()
        input_embeddings = self.embedding.inference([input_ids])
        input_embeddings = input_embeddings[:, : input_ids.shape[1], :]

        # Create multimodal embeddings and attention mask
        multimodal_embeddings = np.concatenate(
            [
                input_embeddings[:, :1, :],
                projected_patch_embeddings,
                input_embeddings[:, 1:, :],
            ],
            axis=1,
        )
        multimodal_attention_mask = torch.cat(
            [
                attention_mask[:, :1],
                projected_patch_attention_mask,
                attention_mask[:, 1:],
            ],
            dim=1,
        )

        # Pad to target sequence length
        current_seq_len = multimodal_embeddings.shape[1]
        padding_len = target_seq_len - current_seq_len
        original_effective_seq_len = current_seq_len

        if padding_len > 0:
            # Get pad embedding
            pad_token_id = 32000
            pad_tensor = torch.tensor(
                [[pad_token_id]], device=attention_mask.device
            )
            pad_emb = self.embedding.inference([pad_tensor.numpy()])
            pad_emb = pad_emb[:, :1, :]

            # Expand pad_emb to match batch size and repeat for padding_len
            pad_embeddings = np.tile(
                pad_emb, (multimodal_embeddings.shape[0], padding_len, 1)
            )
            multimodal_embeddings = np.concatenate(
                [multimodal_embeddings, pad_embeddings], axis=1
            )

            # Create padding for attention mask
            pad_attention_mask = torch.zeros(
                multimodal_attention_mask.shape[0],
                padding_len,
                dtype=multimodal_attention_mask.dtype,
                device=multimodal_attention_mask.device,
            )
            multimodal_attention_mask = torch.cat(
                [multimodal_attention_mask, pad_attention_mask], dim=1
            )

        # Create position IDs
        effective_position_ids = torch.arange(
            original_effective_seq_len,
            dtype=torch.long,
            device=attention_mask.device,
        )
        pad_position_ids = torch.full(
            (padding_len,),
            fill_value=original_effective_seq_len - 1,
            dtype=torch.long,
            device=attention_mask.device,
        )
        sequence_position_ids = torch.cat(
            [effective_position_ids, pad_position_ids], dim=0
        )
        multimodal_position_ids = sequence_position_ids.unsqueeze(0).expand(
            multimodal_embeddings.shape[0], -1
        )

        # Prefill stage
        multimodal_attention_mask = multimodal_attention_mask.numpy()
        multimodal_position_ids = multimodal_position_ids.numpy()
        prefill_input_list = [
            multimodal_attention_mask,
            multimodal_position_ids,
            multimodal_embeddings,
        ]
        prefill_outputs_ptr = self.prefill_model.execute(prefill_input_list, output_to_host=False)
        logits = self.prefill_model.copy_output_to_host(0, prefill_outputs_ptr[0])
        past_kv_ptr = prefill_outputs_ptr[1]

        # Get the last valid token position and extract logits
        last_valid_token_indices = np.sum(multimodal_attention_mask, axis=1) - 1
        logits_last_valid_token = logits[
            np.arange(logits.shape[0]), last_valid_token_indices, :
        ]
        next_token_id = np.argmax(logits_last_valid_token, axis=-1)[:, None]
        generated_ids = next_token_id

        # Decode loop
        for i in range(self.max_new_tokens - 1):
            # Prepare decoder inputs: use only one valid token (matching export shape)
            decoder_input_ids = next_token_id.astype(np.int64)

            # Update attention mask: set the position for the new token
            multimodal_attention_mask[:, original_effective_seq_len + i - 1] = 1
            # Add one new mask segment for the new token
            new_mask = np.array([[1]], dtype=np.int64)
            dummy_attention_mask = np.concatenate(
                [multimodal_attention_mask, new_mask], axis=1
            )

            # Position ID for the single new token
            new_pos_id = np.array(
                [[original_effective_seq_len + i]], dtype=np.int64
            )

            cache_index = np.array([[original_effective_seq_len + i]], dtype=np.int64)
            decoder_inputs = [
                decoder_input_ids,
                dummy_attention_mask,
                new_pos_id,
                past_kv_ptr,
                cache_index,
            ]
            decoder_outputs_ptr = self.decode_model.execute(decoder_inputs, output_to_host=False)
            logits = self.decode_model.copy_output_to_host(0, decoder_outputs_ptr[0])
            past_kv_ptr = decoder_outputs_ptr[1]

            # Extract next token: use logits[:, 0, :] (single token output)
            next_token_id = np.argmax(logits[:, 0, :], axis=-1)[:, np.newaxis]
            generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)

        return generated_ids

    @utils.display_time
    def post_process(self, generated_ids):
        """
        Post-process generated token IDs to obtain actions.

        Args:
            generated_ids: Generated token IDs (batch_size, seq_len)

        Returns:
            Post-processed actions
        """
        return post_process_action(
            generated_ids=generated_ids,
            action_dim=self.action_dim,
            vocab_size=self.vocab_size,
            bin_centers=self.bin_centers,
            action_norm_stats=self.action_norm_stats,
        )


def create_openvla_om_model(config: OpenVLAConfig) -> OpenVLA:
    """
    Create and initialize OpenVLA OM model with all required components.

    Args:
        config: OpenVLA configuration containing all necessary parameters

    Returns:
        Initialized OpenVLA instance and ACL resource
    """
    # Initialize ACL resources
    logging.info("Initializing ACL resources...")
    acl_resource = AclLiteResource(device_id=config.device_id)
    acl_resource.init()

    # Initialize vision models
    logging.info("Loading vision models...")
    vision_backbone = VisionModel(
        config.model_paths.vision_backbone_path,
        config.model_dims.width,
        config.model_dims.height,
    )
    ret = vision_backbone.init()
    utils.check_ret("vision_backbone.init", ret)

    projector = VisionModel(
        config.model_paths.projector_path,
        config.model_dims.width,
        config.model_dims.height,
    )
    ret = projector.init()
    utils.check_ret("projector.init", ret)

    embedding = VisionModel(
        config.model_paths.embedding_path,
        config.model_dims.width,
        config.model_dims.height,
    )
    ret = embedding.init()
    utils.check_ret("embedding.init", ret)

    # Initialize OpenVLA
    logging.info("Loading OpenVLA models...")
    openvla = OpenVLA(
        model_paths=config.model_paths,
        vision_backbone=vision_backbone,
        projector=projector,
        embedding=embedding,
        action_config=config.action_config,
    )
    ret = openvla.init()
    utils.check_ret("openvla.init", ret)

    logging.info("OpenVLA OM model initialization complete.")
    return openvla, acl_resource


def predict_action_with_om(openvla, inputs, target_seq_len: int = 288):
    """
    Predict action using OpenVLA OM models.

    Args:
        openvla: OpenVLA instance with initialized OM models
        inputs: Dictionary containing input_ids, pixel_values, attention_mask
        target_seq_len: Target sequence length for padding (default: 288)

    Returns:
        Predicted actions as numpy array
    """
    input_ids, pixel_values, attention_mask = openvla.pre_process(inputs)
    om_generated_ids = openvla.inference(
        input_ids, pixel_values, attention_mask, target_seq_len=target_seq_len
    )
    om_action = openvla.post_process(om_generated_ids)
    return om_action