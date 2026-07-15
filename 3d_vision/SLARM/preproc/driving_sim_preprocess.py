import glob
import json
import logging
import os

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_cams(cam_file):
    """Load camera-to-world matrices from a trajectory text file."""
    c2w = []
    with open(cam_file, 'r') as cam_fh:
        line = cam_fh.readline()
        while line and line.strip():
            matrix = [float(x) for x in line.strip().split(' ')]
            matrix = np.array(matrix).reshape((4, 4))
            c2w.append(matrix)

            line = cam_fh.readline()

    c2w = np.stack(c2w, 0)  # [::-1]
    return c2w


# Module-level configuration constants
DATA_ROOT = 'data/SLARM_data'
DATASET_NAME = 'driving_sim'
DATASETS_PATH = 'training'
CAM_LIST = ["front_left", "front", "front_right",
            "back_right", "back", "back_left"]
DATASET_FREQUENCY = 20


def main():
    """Main preprocessing function for DrivingSim dataset."""
    source_path = os.path.join(DATA_ROOT, 'datasets', DATASET_NAME, DATASETS_PATH)
    scene_list = os.listdir(source_path)

    for scene in scene_list:
        frame_path = os.path.join(source_path, scene, 'rgb_front_left')
        frame_nums = len(os.listdir(frame_path))
        if frame_nums <= 0:
            raise ValueError(f'No frames found in {frame_path}')

        annotations = {}
        annotations['dataset'] = DATASET_NAME
        annotations['scene_id'] = int(scene)
        annotations['scene_name'] = scene
        annotations['num_timesteps'] = frame_nums
        annotations['camera_list'] = CAM_LIST
        annotations['normalized_time'] = [
            i / DATASET_FREQUENCY for i in range(frame_nums)
        ]

        normalized_intrinsics_dict = {}
        camera_to_world_dict = {}
        camera_to_ego_dict = {}
        original_image_size_dict = {}
        relative_image_path_dict = {}
        for cam in CAM_LIST:
            camera_paras_path = os.path.join(
                source_path, scene, f'camera_{cam}', 'camera.yaml'
            )
            with open(camera_paras_path, 'r') as cam_cfg_fh:
                cam_info = yaml.safe_load(cam_cfg_fh)
                camera_paras = np.array(tuple(map(float, [
                    cam_info['K']['data'][0],
                    cam_info['K']['data'][4],
                    cam_info['K']['data'][2],
                    cam_info['K']['data'][5],
                ])))
                width, height = cam_info['Imagesize']
                original_image_size_dict[cam] = [height, width]
                normalized_intrinsics = np.array([
                    camera_paras[0] / width,
                    camera_paras[1] / height,
                    camera_paras[2] / width,
                    camera_paras[3] / height,
                ])
                normalized_intrinsics_dict[cam] = normalized_intrinsics.tolist()
                transform_data = cam_info['Transformation']['data']
                camera_to_ego = np.array(transform_data).reshape(4, 4).tolist()
                camera_to_ego_dict[cam] = camera_to_ego

            trj_txt_filepath = glob.glob(
                os.path.join(source_path, scene, f'camera_{cam}', 'trj_*')
            )
            trj_txt_filename = trj_txt_filepath[0].split('/')[-1]
            camera_trajs_path = os.path.join(
                source_path, scene, f'camera_{cam}', trj_txt_filename
            )
            c2ws = load_cams(camera_trajs_path)

            tmp = []
            tmp2 = []
            for frame_idx in range(frame_nums):
                tmp.append(c2ws[frame_idx].tolist())
                tmp2.append(os.path.join(
                    DATASETS_PATH, scene, f'rgb_{cam}', f'{frame_idx:05d}.jpg'
                ))
            camera_to_world_dict[cam] = tmp
            relative_image_path_dict[cam] = tmp2

        annotations['normalized_intrinsics'] = normalized_intrinsics_dict
        annotations['camera_to_world'] = camera_to_world_dict
        annotations['camera_to_ego'] = camera_to_ego_dict
        annotations['original_image_size'] = original_image_size_dict
        annotations['relative_image_path'] = relative_image_path_dict
        annotations['fps'] = DATASET_FREQUENCY

        save_dir = os.path.join(
            DATA_ROOT, 'annotations', DATASET_NAME, DATASETS_PATH
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{scene}.json')
        with open(save_path, "w") as f:
            json.dump(annotations, f)
        logger.info('Saving: %s', save_path)

    logger.info('done')


if __name__ == "__main__":
    main()

