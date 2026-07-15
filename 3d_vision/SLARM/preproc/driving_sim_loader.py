import logging
import os
from argparse import ArgumentParser

import cv2
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_rgb_data(file_path):
    data = cv2.imread(file_path)
    logger.info("rgb: shape %s, min %s, max %s", data.shape, data.min(), data.max())
    return data


def load_depth_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    data_valid_mask = data < 65535  # data is clipped at 65535, i.e. 655.35m
    data = data.astype(np.float32) / 100.
    logger.info("depth: shape %s, min %s, max %s", data.shape, data.min(), data.max())
    return data, data_valid_mask


def load_normal_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    data = data.astype(np.float32)
    data = data / 255 * 2 - 1
    logger.info("normal: shape %s, min %s, max %s", data.shape, data.min(), data.max())
    return data


def load_semantic_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    logger.info("semantic: shape %s, min %s, max %s", data.shape, data.min(), data.max())
    return data


def load_instance_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    cls_data = data[..., 0]
    inst_data = data[..., 1:].astype(np.uint16)
    inst_data = inst_data[..., 0] * 256 + inst_data[..., 1]
    logger.info("semantic (from instance): shape %s, min %s, max %s", cls_data.shape, cls_data.min(), cls_data.max())
    logger.info("instance: shape %s, min %s, max %s", inst_data.shape, inst_data.min(), inst_data.max())
    return cls_data, inst_data


def load_scene_flow_data(file_path):
    data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    data = data.astype(np.float32)
    data = (data / 65535 * 2 - 1) * 5  # Unit: meters
    data = data * 20  # 20Hz -> Unit: m/s
    # (+x: front, +y: left, +z: up)
    logger.info("scene flow: shape %s, min %s, max %s", data.shape, data.min(), data.max())
    logger.info("X-direction velocity: min = %.4f m/s, max = %.4f m/s", data[..., 0].min(), data[..., 0].max())
    logger.info("Y-direction velocity: min = %.4f m/s, max = %.4f m/s", data[..., 1].min(), data[..., 1].max())
    logger.info("Z-direction velocity: min = %.4f m/s, max = %.4f m/s", data[..., 2].min(), data[..., 2].max())
    return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--data_folder', type=str, required=True,
                        help="Path to the driving_sim scene folder, e.g., "
                             "data/SLARM_data/datasets/driving_sim/training/040")
    args = parser.parse_args()
    data_folder = args.data_folder

    # Validate path exists
    if not os.path.exists(data_folder):
        logger.error("Data folder does not exist: %s", data_folder)
        logger.error("Please provide a valid path using -i or --data_folder argument")
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    rgb_path = os.path.join(data_folder, "rgb_front", f"{0:05d}.jpg")
    load_rgb_data(rgb_path)

    depth_path = os.path.join(data_folder, "depth_front", f"{0:05d}.png")
    load_depth_data(depth_path)

    normal_path = os.path.join(data_folder, "normal_front", f"{0:05d}.png")
    load_normal_data(normal_path)

    semantic_path = os.path.join(data_folder, "semantic_front", f"{0:05d}.png")
    load_semantic_data(semantic_path)

    instance_path = os.path.join(data_folder, "instance_front", f"{0:05d}.png")
    load_instance_data(instance_path)

    scene_flow_path = os.path.join(data_folder, "scene_flow_front", f"00001.png")
    load_scene_flow_data(scene_flow_path)
