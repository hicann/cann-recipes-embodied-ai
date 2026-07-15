# coding=utf-8
# Adapted from
# https://github.com/AI-Hypercomputer/gpu-recipes/blob/main/src/utils/data_processing/waymo_dataset/waymo_perception_data_processor.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Waymo Open Dataset is provided by Waymo LLC under the Waymo Dataset License Agreement.
# See https://waymo.com/open/terms/ for dataset terms and conditions.

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

opencv2waymo = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

# flu(forward-left-up) coordinate system: x-forward y-left z-up
DATASETS = {
    "driving_sim": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
    "waymo": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    "nuscenes": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
    "b2d": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
}

waymo_train = "scene_list/waymo_train.txt"  # NOTE: Use full data for multi-GPU
waymo_val = "scene_list/waymo_val.txt"
driving_sim_train = "scene_list/driving_sim_train.txt"
driving_sim_val = "scene_list/driving_sim_val.txt"
b2d_train = "scene_list/b2d_train.txt"
b2d_val = "scene_list/b2d_val.txt"

if os.environ.get('OVERFIT_EXP'):  # Overfitting experiment on a single dynamic scene
    # b2d scene id TBD, modify b2d_train_1.txt directly for overfitting experiment
    b2d_train = "scene_list/b2d_train_1.txt"
    b2d_val = "scene_list/b2d_train_1.txt"
    logger.info("B2D overfitting using default scene")

    # Specify waymo scene id
    if os.environ.get('SCENE_ID_WAYMO'):
        scene_list_file_path = "data/dataset_scene_list/waymo_train_list.txt"
        if os.environ.get('USE_VALIDATION'):
            scene_list_file_path = "data/dataset_scene_list/waymo_val_list.txt"
        try:
            with open(scene_list_file_path, 'r', encoding='utf-8') as file:
                # Read all lines and remove trailing newlines
                lines = [line.rstrip('\n') for line in file]
            scene_id = int(os.environ.get('SCENE_ID_WAYMO'))
            scene_name = lines[scene_id].strip()
            scene_file_context = f"annotations/waymo/training/{scene_name}.json"
            if os.environ.get('USE_VALIDATION'):
                scene_file_context = f"annotations/waymo/validation/{scene_name}.json"
            scene_file_name = "scene_list/waymo_train_1_scene.txt"
            with open(os.path.join(os.environ.get('DATA_ROOT').strip(), scene_file_name),
                      'w', encoding='utf-8') as file:
                file.write(scene_file_context)
            waymo_train = scene_file_name
            waymo_val = scene_file_name
            logger.info("Waymo training on Scene %s.", scene_id)
        except FileNotFoundError:
            logger.warning("File '%s' does not exist, using default scene", scene_list_file_path)
            waymo_train = "scene_list/waymo_train_1.txt"
            waymo_val = "scene_list/waymo_train_1.txt"
        except Exception as e:
            logger.warning("Error determining Waymo scene: %s, using default scene", e)
            waymo_train = "scene_list/waymo_train_1.txt"
            waymo_val = "scene_list/waymo_train_1.txt"
    else:
        waymo_train = "scene_list/waymo_train_1.txt"
        waymo_val = "scene_list/waymo_train_1.txt"
        logger.info("Waymo overfitting ID not specified, using default scene")

    # Specify driving_sim scene id
    if os.environ.get('SCENE_ID_DRIVING_SIM'):
        try:
            scene_id = int(os.environ.get('SCENE_ID_DRIVING_SIM'))
            # Convert to 3-digit string, padded with leading zeros
            scene_name = f"{scene_id:03d}"
            scene_file_context = f"annotations/driving_sim/training/{scene_name}.json"
            if os.environ.get('USE_VALIDATION'):
                scene_file_context = f"annotations/driving_sim/validation/{scene_name}.json"
            scene_file_name = "scene_list/driving_sim_train_1_scene.txt"
            with open(os.path.join(os.environ.get('DATA_ROOT').strip(), scene_file_name),
                      'w', encoding='utf-8') as file:
                file.write(scene_file_context)
            driving_sim_train = scene_file_name
            driving_sim_val = scene_file_name
            logger.info("Driving_sim training on Scene %s.", scene_id)
        except Exception as e:
            logger.warning("Error determining Driving_sim scene: %s, using default scene", e)
            driving_sim_train = "scene_list/driving_sim_train_1.txt"
            driving_sim_val = "scene_list/driving_sim_train_1.txt"
    else:
        driving_sim_train = "scene_list/driving_sim_train_1.txt"
        driving_sim_val = "scene_list/driving_sim_train_1.txt"
        logger.info("Driving_sim overfitting ID not specified, using default scene")

# Shared camera list templates to avoid duplication across datasets
_CAM_LIST_3 = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]

_CAM_LIST_5 = [
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
]

_CAM_LIST_6 = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

_CAM_LIST_7 = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

DATASET_DICT = {
    "b2d": {
        "size": [160, 240],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": b2d_train,
        "annotation_txt_file_val": b2d_val,
        "camera_list": {
            1: ["CAM_FRONT"],
            3: _CAM_LIST_3,
            5: _CAM_LIST_5,
            6: _CAM_LIST_6,
            7: _CAM_LIST_7,
        },
        "ref_camera": "CAM_FRONT",
    },

    "driving_sim": {
        "size": [160, 240],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": driving_sim_train,
        "annotation_txt_file_val": driving_sim_val,
        "camera_list": {
            1: ["front"],
            3: ["front_left", "front", "front_right"],
            5: ["back_left", "front_left", "front", "front_right", "back_right"],
            6: ["front_left", "front", "front_right", "back_right", "back", "back_left"],
            7: ["front_left", "front", "front_right", "back_right", "back", "back_left"],
        },
        "ref_camera": "front",
    },

    "waymo": {
        "size": [160, 240],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": waymo_train,
        "annotation_txt_file_val": waymo_val,
        "camera_list": {
            1: ["0"],
            3: ["1", "0", "2"],
            5: ["3", "1", "0", "2", "4"],
            6: ["3", "1", "0", "2", "4"],  # capped at 5
            7: ["3", "1", "0", "2", "4"],
        },
        "ref_camera": "0",
    },
    "nuscenes": {
        "size": [160, 288],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": "scene_list/nuscenes_train.txt",
        "annotation_txt_file_val": "scene_list/nuscenes_val.txt",
        "camera_list": {
            1: ["CAM_FRONT"],
            3: _CAM_LIST_3,
            5: _CAM_LIST_5,
            6: _CAM_LIST_6,
            7: _CAM_LIST_7,
        },
        "ref_camera": "CAM_FRONT",
    },
    "rel10k": {
        "size": [160, 296],
        "temporal": False,
        "num_context_timesteps": 2,
        "num_target_timesteps": 8,
        "annotation_txt_file_train": "scene_list/rel10k_train.txt",
        "annotation_txt_file_val": "scene_list/rel10k_val.txt",
        "batch_size_scale": 6,
        "camera_list": {
            1: ["0"],
            3: ["0"],
            5: ["0"],
            6: ["0"],
            7: ["0"],
        },
    },
    "dl3dv": {
        "size": [160, 288],
        "temporal": False,
        "num_context_timesteps": 2,
        "num_target_timesteps": 8,
        "annotation_txt_file_train": "scene_list/dl3dv_train.txt",
        "annotation_txt_file_val": "scene_list/dl3dv_val.txt",
        "batch_size_scale": 6,
        "camera_list": {
            1: ["0"],
            3: ["0"],
            5: ["0"],
            6: ["0"],
            7: ["0"],
        },
    },
}

FEAT_TYPES = ["pe3r", "lseg"]
ONLINE_FEAT_TYPES = ["lseg"]

# text label list
# NOTE: If you change this category, all the corresponding categories below it
# must also be changed. For example WAYMO_SEMANTIC_ID_TO_LABEL,
# B2D_SEMANTIC_ID_TO_LABEL, DRIVING_SIM_SEMANTIC_ID_TO_LABEL
SEMANTIC_LABEL_LIST = [
    'others', 'vehicle', 'bicycle', 'motorcycle', 'people',
    'buildings', 'road', 'sidewalk', 'sky',
]

# text label to idx
SEMANTIC_LABEL_TO_IDX = {}
for i, label in enumerate(SEMANTIC_LABEL_LIST):
    SEMANTIC_LABEL_TO_IDX[label] = i

SEMANTIC_ID_TO_COLOR = [
    [99., 178., 50.],  # others
    [199., 2., 14.],  # vehicle
    [203., 56., 122.],  # bicycle
    [153., 56, 58.],  # motorcycle
    [252., 200., 0.],  # people
    [128., 128., 128.],  # buildings
    [50., 181., 199.],  # road
    [236., 110., 3.],  # sidewalk
    [34., 121., 182.],  # sky
]


def textlabel_to_id(semantic_id_to_label):
    """Map dataset-specific semantic IDs to unified text label IDs.

    Returns a dict of {text_label: [semantic_id, ...]}.
    """
    semantic_label_to_id = {}
    for key, value in semantic_id_to_label.items():
        if value not in SEMANTIC_LABEL_LIST:
            raise ValueError(f"Class [{value}] not in SEMANTIC_LABEL_LIST!")
        if value not in semantic_label_to_id:
            semantic_label_to_id[value] = [key]
        else:
            semantic_label_to_id[value].append(key)
    return semantic_label_to_id


def id_to_idx(semantic_id_to_label):
    """Map dataset-specific semantic IDs to unified label indices.

    Returns a dict of {semantic_id: label_idx}.
    """
    semantic_id_to_idx = {}
    for key, value in semantic_id_to_label.items():
        semantic_id_to_idx[key] = SEMANTIC_LABEL_TO_IDX[value]
    return semantic_id_to_idx


WAYMO_SEMANTIC_CLASSES = {
    # Anything that does not fit the other classes or is too ambiguous to label.
    "TYPE_UNDEFINED": 0,
    "TYPE_EGO_VEHICLE": 1,  # The Waymo vehicle.
    # Small vehicle such as a sedan, SUV, pickup truck, minivan or golf cart.
    "TYPE_CAR": 2,
    "TYPE_TRUCK": 3,  # Large vehicle that carries cargo.
    "TYPE_BUS": 4,  # Large vehicle that carries more than 8 passengers.
    # Large vehicle that is not a truck or a bus.
    "TYPE_OTHER_LARGE_VEHICLE": 5,
    "TYPE_BICYCLE": 6,  # Bicycle with no rider.
    "TYPE_MOTORCYCLE": 7,  # Motorcycle with no rider.
    # Trailer attached to another vehicle or horse.
    "TYPE_TRAILER": 8,
    # Pedestrian. Does not include objects associated with the pedestrian,
    # such as suitcases, strollers or cars.
    "TYPE_PEDESTRIAN": 9,
    "TYPE_CYCLIST": 10,  # Bicycle with rider.
    "TYPE_MOTORCYCLIST": 11,  # Motorcycle with rider.
    # Birds, including ones on the ground.
    "TYPE_BIRD": 12,
    # Animal on the ground such as a dog, cat, cow, etc.
    "TYPE_GROUND_ANIMAL": 13,
    # Cone or short pole related to construction.
    "TYPE_CONSTRUCTION_CONE_POLE": 14,
    # Permanent horizontal and vertical lamp pole, traffic sign pole, etc.
    "TYPE_POLE": 15,
    # Large object carried/pushed/dragged by a pedestrian.
    "TYPE_PEDESTRIAN_OBJECT": 16,
    # Sign related to traffic, including front and back facing signs.
    "TYPE_SIGN": 17,
    # The box that contains traffic lights regardless of front or back facing.
    "TYPE_TRAFFIC_LIGHT": 18,
    # Permanent building and walls, including solid fences.
    "TYPE_BUILDING": 19,
    # Drivable road with proper markings, including parking lots and gas stations.
    "TYPE_ROAD": 20,
    # Marking on the road that is parallel to the ego vehicle and defines lanes.
    "TYPE_LANE_MARKER": 21,
    # All markings on the road other than lane markers.
    "TYPE_ROAD_MARKER": 22,
    # Paved walkable surface for pedestrians, including curbs.
    "TYPE_SIDEWALK": 23,
    # Vegetation including tree trunks, tree branches, bushes, tall grasses,
    # flowers and so on.
    "TYPE_VEGETATION": 24,
    "TYPE_SKY": 25,  # The sky, including clouds.
    # Other horizontal surfaces that are drivable or walkable.
    "TYPE_GROUND": 26,
    # Object that is not permanent in its current position and does not belong
    # to any of the above classes.
    "TYPE_DYNAMIC": 27,
    # Object that is permanent in its current position and does not belong
    # to any of the above classes.
    "TYPE_STATIC": 28,
}

WAYMO_SEMANTIC_ID_TO_LABEL = {
    0: "others",
    1: "others",
    2: "vehicle",
    3: "vehicle",
    4: "vehicle",
    5: "vehicle",
    6: "bicycle",
    7: "motorcycle",
    8: "others",
    9: "people",
    10: "bicycle",
    11: "motorcycle",
    12: "others",
    13: "others",
    14: "others",
    15: "others",
    16: "others",
    17: "others",
    18: "others",
    19: "buildings",
    20: "road",
    21: "road",  # lane marker
    22: "road",  # road_marker
    23: "sidewalk",
    # vegetation: tree trunks, tree branches, bushes, tall grasses, flowers, etc.
    24: "others",
    25: "sky",
    # ground: Other horizontal surfaces that are drivable or walkable.
    26: "road",
    # dynamic: Object not permanent in its current position
    27: "others",
    # static: Object permanent in its current position
    28: "others",
}

# Semantic classes for the camera segmentation labels.
B2D_SEMANTIC_CLASSES = {
    # Elements that have not been categorized are considered Unlabeled.
    "Unlabeled": 0,
    # Part of ground on which cars usually drive.
    "Roads": 1,
    # Part of ground designated for pedestrians or cyclists. Delimited from the
    # road by some obstacle (such as curbs or poles), not only by markings.
    # This label includes a possibly delimiting curb, traffic islands
    # (the walkable part), and pedestrian zones.
    "SideWalks": 2,
    # Buildings like houses, skyscrapers,... and the elements attached to them.
    # E.g. air conditioners, scaffolding, awning or ladders and much more.
    "Building": 3,
    # Individual standing walls. Not part of a building.
    "Wall": 4,
    # Barriers, railing, or other upright structures. Basically wood or wire
    # assemblies that enclose an area of ground.
    "Fence": 5,
    # Small mainly vertically oriented pole. If the pole has a horizontal part
    # (often for traffic light poles) this is also considered pole.
    # E.g. sign pole, traffic light poles.
    "Pole": 6,
    # Traffic light boxes without their poles.
    "TrafficLight": 7,
    # Signs installed by the state/city authority, usually for traffic
    # regulation. This category does not include the poles where signs are
    # attached to. E.g. traffic- signs, parking signs, direction signs...
    "TrafficSign": 8,
    # Trees, hedges, all kinds of vertical vegetation. Ground-level vegetation
    # is considered Terrain.
    "Vegetation": 9,
    # Grass, ground-level vegetation, soil or sand. These areas are not meant
    # to be driven on. This label includes a possibly delimiting curb.
    "Terrain": 10,
    "Sky": 11,  # Open sky. Includes clouds and the sun.
    "Pedestrian": 12,  # Humans that walk
    # Humans that ride/drive any kind of vehicle or mobility system
    # E.g. bicycles or scooters, skateboards, horses, roller-blades,
    # wheel-chairs, etc.
    "Rider": 13,
    "Car": 14,  # Cars, vans
    "Truck": 15,  # Trucks
    "Bus": 16,  # Busses
    "Train": 17,  # Trains
    "Motorcycle": 18,  # Motorcycle, Motorbike
    "Bicycle": 19,  # Bicylces
    # Elements in the scene and props that are immovable.
    # E.g. fire hydrants, fixed benches, fountains, bus stops, etc.
    "Static": 20,
    # Elements whose position is susceptible to change over time.
    # E.g. Movable trash bins, buggies, bags, wheelchairs, animals, etc.
    "Dynamic": 21,
    # Everything that does not belong to any other category.
    "Other": 22,
    # Horizontal water surfaces. E.g. Lakes, sea, rivers.
    "Water": 23,
    "RoadLine": 24,  # The markings on the road.
    # Any horizontal ground-level structures that does not match any other
    # category. For example areas shared by vehicles and pedestrians, or flat
    # roundabouts delimited from the road by a curb.
    "Ground": 25,
    # Only the structure of the bridge. Fences, people, vehicles, an other
    # elements on top of it are labeled separately.
    "Bridge": 26,
    # All kind of rail tracks that are non-drivable by cars.
    # E.g. subway and train rail tracks.
    "RailTrack": 27,
    "GuardRail": 28,  # All types of guard rails/crash barriers.
}

B2D_SEMANTIC_ID_TO_LABEL = {
    0: "others",
    1: "road",
    2: "sidewalk",
    3: "buildings",
    4: "others",
    5: "others",
    6: "others",
    7: "others",
    8: "others",
    9: "others",
    10: "others",
    11: "sky",
    12: "people",
    13: "people",
    14: "vehicle",
    15: "vehicle",
    16: "vehicle",
    17: "vehicle",
    18: "motorcycle",
    19: "bicycle",
    20: "others",
    21: "others",
    22: "others",
    23: "others",
    24: "road",
    25: "sidewalk",
    26: "others",
    27: "others",
    28: "others",
}

# Semantic classes for the camera segmentation labels.
DRIVING_SIM_SEMANTIC_CLASSES = {
    "building": 0,
    "road": 1,
    "car": 2,  # car/truck
    "sidewalk": 3,  # sidewalk/sideway
    "tree": 4,
    "bench": 5,
    "signboard": 6,
    "hydrant": 7,
    "streetlamp": 8,
    "trafficlight": 9,
    "freeway": 10,
    "trashcan": 11,
    "fence": 12,  # fence/guardrail
    "mailbox": 13,
    "parkingmeter": 14,
    "buststop": 15,
    "trafficsign": 16,
    "other": 17,
    "sky": 18,
    "ground": 19,
    "pole": 20,
    "vegetation": 21,
    "person": 22,
    "trafficline": 23,
    "dynamic_car": 24,
}

DRIVING_SIM_SEMANTIC_ID_TO_LABEL = {
    0: "buildings",
    1: "road",
    2: "vehicle",
    3: "sidewalk",
    4: "others",
    5: "others",
    6: "others",
    7: "others",
    8: "others",
    9: "others",
    10: "road",
    11: "others",
    12: "others",
    13: "others",
    14: "others",
    15: "others",
    16: "others",
    17: "others",
    18: "sky",
    19: "road",
    20: "others",
    21: "others",
    22: "people",
    23: "others",
    24: "vehicle",
}

# Datasets mapping: text label to id
WAYMO_SEMANTIC_LABEL_TO_ID = textlabel_to_id(WAYMO_SEMANTIC_ID_TO_LABEL)
B2D_SEMANTIC_LABEL_TO_ID = textlabel_to_id(B2D_SEMANTIC_ID_TO_LABEL)
DRIVING_SIM_SEMANTIC_LABEL_TO_ID = textlabel_to_id(DRIVING_SIM_SEMANTIC_ID_TO_LABEL)

# Datasets mapping: id to idx
WAYMO_SEMANTIC_ID_TO_IDX = id_to_idx(WAYMO_SEMANTIC_ID_TO_LABEL)
B2D_SEMANTIC_ID_TO_IDX = id_to_idx(B2D_SEMANTIC_ID_TO_LABEL)
DRIVING_SIM_SEMANTIC_ID_TO_IDX = id_to_idx(DRIVING_SIM_SEMANTIC_ID_TO_LABEL)

SEMANTIC_ID_TO_IDX_DICT = {
    'waymo': WAYMO_SEMANTIC_ID_TO_IDX,
    'b2d': B2D_SEMANTIC_ID_TO_IDX,
    'driving_sim': DRIVING_SIM_SEMANTIC_ID_TO_IDX,
}
