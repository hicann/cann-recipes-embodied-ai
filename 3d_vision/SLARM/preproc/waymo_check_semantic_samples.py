import logging
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


F_PATH = 'xxx/SLARM_data/datasets/waymo/training'
SCENES = 798

num_scenes_all = 0
num_samples_all = 0
for scene_id in range(SCENES):
    sub_fpath = os.path.join(F_PATH, str(scene_id).zfill(3), 'semantic_segs')
    labels_num = len(os.listdir(sub_fpath))
    if labels_num > 0:
        num_scenes_all += 1
        num_samples_all += labels_num

logger.info('%s', num_scenes_all)
logger.info('%s', num_samples_all)
