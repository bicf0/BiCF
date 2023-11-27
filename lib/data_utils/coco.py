import os
import cv2
import glob
import h5py
import json
import joblib
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import scipy.io as sio


from lib.models import spin
from lib.core.config import BLVF_DATA_DIR, BLVF_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import get_bbox_from_kp2d
from lib.data_utils.feature_extractor import extract_features


def read_data_train(dataset_path, debug=False):
    dataset = {
        ''
    }













if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/blvf_hmr_db/coco')
    args = parser.parse_args()

    dataset = read_data_train(args.dir)
    joblib.dump(dataset, osp.join(BLVF_DB_DIR, 'coco_train_db.pt'))