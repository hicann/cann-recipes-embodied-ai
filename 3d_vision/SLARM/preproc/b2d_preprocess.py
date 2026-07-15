import glob
import gzip
import json
import logging
import os

import cv2
import numpy as np
import open3d as o3d  # in docker
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

stand_to_ue4_rotate = np.array([[0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1, 1] = -1


def convert_extrinsic_4x4_left_to_right(t_left):
    return np.linalg.inv(stand_to_ue4_rotate) @ t_left @ left2right


def load_json_gz(file):
    with gzip.open(file, 'rt', encoding='utf-8') as gz_file:
        anno_ = json.load(gz_file)
        gz_file.close()
    return anno_


b2d_camera_tags2path_name = {
    'CAM_FRONT': 'rgb_front',
    'CAM_FRONT_LEFT': 'rgb_front_left',
    'CAM_FRONT_RIGHT': 'rgb_front_right',
    'CAM_BACK': 'rgb_back',
    'CAM_BACK_LEFT': 'rgb_back_left',
    'CAM_BACK_RIGHT': 'rgb_back_right',
}

DATA_ROOT = 'data/SLARM_data'
DATASET_NAME = 'b2d'
DATASETS_PATH = 'training'
cam_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
DATASET_FREQUENCY = 10


def main():
    """Main preprocessing function for Bench2Drive dataset."""
    source_path = os.path.join(DATA_ROOT, 'datasets', DATASET_NAME, DATASETS_PATH)
    scene_list = os.listdir(source_path)
    scene_list = [s for s in scene_list if not s.endswith('.tar')]  # there are tar files in the directory

    for scene_idx, scene in enumerate(scene_list):
        frame_nums = len(os.listdir(os.path.join(source_path, scene, "anno")))

        if frame_nums == 0:
            continue

        annotations = {}
        annotations['dataset'] = DATASET_NAME
        annotations['scene_id'] = scene_idx
        annotations['scene_name'] = scene
        annotations['num_timesteps'] = frame_nums
        annotations['camera_list'] = cam_list
        annotations['normalized_time'] = [i / DATASET_FREQUENCY for i in range(frame_nums)]

        normalized_intrinsics_dict = {}
        camera_to_world_dict = {}
        camera_to_ego_dict = {}
        original_image_size_dict = {}
        relative_image_path_dict = {}
        for cam in cam_list:
            frame_idx_str = "{:05d}".format(0)  # first frame
            anno_path = os.path.join(source_path, scene, "anno", f"{frame_idx_str}.json.gz")
            anno = load_json_gz(anno_path)
            intrinsic = np.array(anno['sensors'][cam]['intrinsic'])
            img_size_x = anno['sensors'][cam]['image_size_x']
            img_size_y = anno['sensors'][cam]['image_size_y']
            normalized_intrinsic = [
                intrinsic[0, 0] / img_size_x,
                intrinsic[1, 1] / img_size_y,
                intrinsic[0, 2] / img_size_x,
                intrinsic[1, 2] / img_size_y
            ]
            original_image_size_dict[cam] = [img_size_y, img_size_x]
            normalized_intrinsics_dict[cam] = normalized_intrinsic
            camera_to_ego = np.array(anno['sensors'][cam]['cam2ego']).reshape(4, 4)
            ego_to_camera = np.linalg.inv(camera_to_ego)
            ego_to_camera = convert_extrinsic_4x4_left_to_right(ego_to_camera)
            camera_to_ego_dict[cam] = np.linalg.inv(ego_to_camera).tolist()

            tmp = []
            tmp2 = []
            for frame_idx in range(frame_nums):
                frame_idx_str = "{:05d}".format(frame_idx)
                anno_path = os.path.join(source_path, scene, "anno", f"{frame_idx_str}.json.gz")
                anno = load_json_gz(anno_path)
                world2cam = np.array(anno['sensors'][cam]['world2cam'])
                world2cam = convert_extrinsic_4x4_left_to_right(world2cam)
                c2w = np.linalg.inv(world2cam)
                tmp.append(c2w.tolist())
                relative_image_path = os.path.join(
                    DATASETS_PATH,
                    scene,
                    "camera",
                    b2d_camera_tags2path_name[cam],
                    f"{frame_idx_str}.jpg",
                )
                tmp2.append(relative_image_path)
            camera_to_world_dict[cam] = tmp
            relative_image_path_dict[cam] = tmp2

        annotations['normalized_intrinsics'] = normalized_intrinsics_dict
        annotations['camera_to_world'] = camera_to_world_dict
        annotations['camera_to_ego'] = camera_to_ego_dict
        annotations['original_image_size'] = original_image_size_dict
        annotations['relative_image_path'] = relative_image_path_dict
        annotations['fps'] = DATASET_FREQUENCY

        save_dir = os.path.join(DATA_ROOT, 'annotations', DATASET_NAME, DATASETS_PATH)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{scene}.json')
        with open(save_path, "w") as f:
            json.dump(annotations, f)
        logger.info('Saving: %s', save_path)

    logger.info('done')


if __name__ == "__main__":
    main()

