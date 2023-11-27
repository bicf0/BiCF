# -*- coding:utf-8 -*-

import os
import torch
import logging
import numpy as np
import joblib

from torch.utils.data import Dataset
from lib.core.config_model_data import TRAIN_TEST_DATA
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import normalize_2d_kp, transfrom_keypoints, split_into_chunks

logger = logging.getLogger(__name__)


# train dataset
class MixDataset(Dataset):
    def __init__(self, dataset_name, dataset_weight, seq_len, overlap):
        self.dataset_name = dataset_name
        self.dataset_weight = dataset_weight
        self.seq_len = seq_len
        self.stride = int(seq_len*(1-overlap))
        self.db = self.load_db()
        self.indices = split_into_chunks(self.db['vid_name'], self.seq_len, self.stride)
        # logger.info()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = os.path.join(TRAIN_TEST_DATA, f'{self.dataset_name}/{self.dataset_name}_train_occ_db.pt')
        assert os.path.isfile(db_file), "the file not exists!!!"
        db = joblib.load(db_file)
        logger.info(f'Loaded {self.dataset_name} dataset from {db_file}, and assigned weight {self.dataset_weight}')

        return db

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seq_len, axis=0)  # deal with the last sequence

    def get_single_item(self, index):
        start_index, end_index = self.indices[index]

        if self.dataset_name == 'h36m':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])  # spin (49)
            kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])  # spin (49)

            pose = self.get_sequence(start_index, end_index, self.db['pose'])   # has
            shape = self.get_sequence(start_index, end_index, self.db['shape'])   # has
            w_smpl = torch.ones(self.seq_len).float()   # has == 1
            w_3d = torch.ones(self.seq_len).float()     # has == 1

        elif self.dataset_name == 'mpi_inf_3dhp':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])  # spin (49)
            kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])  # spin (49)

            pose = np.zeros((kp_2d.shape[0], 72))       # all 0
            shape = np.zeros((kp_2d.shape[0], 10))      # all 0
            w_smpl = torch.zeros(self.seq_len).float()  # not_has == 0
            w_3d = torch.ones(self.seq_len).float()     # has == 1

        else:  # 3dpw
            kp_2d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints2D']), src='common',
                                dst='spin')                                              # spin 49)
            kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])      # spin (49)

            pose = self.get_sequence(start_index, end_index, self.db['pose'])
            shape = self.get_sequence(start_index, end_index, self.db['shape'])
            w_smpl = torch.ones(self.seq_len).float()
            w_3d = torch.ones(self.seq_len).float()

        # feature = np.ones((self.seq_len, 49, 2), dtype=np.float16)
        kp_2d_tensor = np.ones((self.seq_len, 49, 3), dtype=np.float16)
        kp_3d_tensor = np.zeros((self.seq_len, 49, 3), dtype=np.float16)
        bbox = self.get_sequence(start_index, end_index, self.db['bbox'])
        feature = torch.from_numpy(self.get_sequence(start_index, end_index, self.db['features'])).float()
        theta_tensor = np.zeros((self.seq_len, 85), dtype=np.float16)

        # for idx in range(self.seq_len):
        #     features[idx] = kp_2d[idx, :, :2]
        #     kp_2d_tensor[idx] = kp_2d[idx]
        #     kp_3d_tensor[idx] = kp_3d[idx]
        #     # theta shape (85,)
        #     theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
        #     theta_tensor[idx] = theta

        # crop image and transform 2d key points
        for idx in range(self.seq_len):
            kp_2d[idx, :, :2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx, :, :2],
                center_x=bbox[idx, 0],
                center_y=bbox[idx, 1],
                width=bbox[idx, 2],
                height=bbox[idx, 3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx, :, :2] = normalize_2d_kp(kp_2d[idx, :, :2], 224)

            kp_2d_tensor[idx] = kp_2d[idx]
            # feature[idx] = kp_2d[idx, :, :2]
            kp_3d_tensor[idx] = kp_3d[idx]
            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
            theta_tensor[idx] = theta

        # print(f"features-shape: {features.shape}")
        target = {
            'features': feature,     # (seq_len, 49, 2)
            'theta': torch.from_numpy(theta_tensor).float(),    # (seq_len, 85)
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),    # (seq_len, 49, 3)
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(),    # (seq_len, 49, 3)
            'w_smpl': w_smpl,                                   # (seq_len,)
            'w_3d': w_3d,                                       # (seq_len,)
        }

        return target


# valid dataset
class ValidDataset(Dataset):
    def __init__(self, dataset_name, seq_len, overlap):
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.stride = int(seq_len*(1-overlap))
        self.db = self.load_db()
        # print(self.stride)
        self.indices = split_into_chunks(self.db['vid_name'], self.seq_len, self.stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = os.path.join(TRAIN_TEST_DATA, f'{self.dataset_name}/{self.dataset_name}_val_db.pt')
        assert os.path.isfile(db_file), "the file not exists!!!"
        db = joblib.load(db_file)
        logger.info(f'Loaded {self.dataset_name} dataset from {db_file}')

        return db

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seq_len, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.indices[index]

        if self.dataset_name == 'h36m':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])          # spin(49) not used when val
            kp_3d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints3D']), src='spin',
                                dst='common')                                               # common(test14) real label

            pose = np.zeros((kp_2d.shape[0], 72))
            shape = np.zeros((kp_2d.shape[0], 10))
            w_smpl = torch.zeros(self.seq_len).float()
            w_3d = torch.ones(self.seq_len).float()

        elif self.dataset_name == 'mpi_inf_3dhp':
            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
            kp_3d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints3D']), src='spin',
                                dst='common')                                          # mpii_test(17)

            pose = np.zeros((kp_2d.shape[0], 72))
            shape = np.zeros((kp_2d.shape[0], 10))
            w_smpl = torch.zeros(self.seq_len).float()
            w_3d = torch.ones(self.seq_len).float()

        else:
            kp_2d = convert_kps(self.get_sequence(start_index, end_index, self.db['joints2D']), src='common',
                                dst='spin')                                                 # spin(49)
            kp_3d = self.get_sequence(start_index, end_index, self.db['joints3D'])          # common(test14) real label

            pose = self.get_sequence(start_index, end_index, self.db['pose'])
            shape = self.get_sequence(start_index, end_index, self.db['shape'])
            w_smpl = torch.ones(self.seq_len).float()
            w_3d = torch.ones(self.seq_len).float()

        # feature = np.ones((self.seq_len, 49, 2), dtype=np.float16)
        kp_2d_tensor = np.ones((self.seq_len, 49, 3), dtype=np.float16)
        # if self.dataset_name == 'mpi_inf_3dhp':
        #     nj = 17
        # else:
        nj = 14
        kp_3d_tensor = np.zeros((self.seq_len, nj, 3), dtype=np.float16)
        bbox = self.get_sequence(start_index, end_index, self.db['bbox'])
        feature = torch.from_numpy(self.get_sequence(start_index, end_index, self.db['features'])).float()
        theta_tensor = np.zeros((self.seq_len, 85), dtype=np.float16)

        # for idx in range(self.seq_len):
        #     features[idx] = kp_2d[idx, :, :2]
        #     kp_2d_tensor[idx] = kp_2d[idx]
        #     kp_3d_tensor[idx] = kp_3d[idx]
        #     # theta shape (85,)
        #     theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
        #     theta_tensor[idx] = theta

        # crop image and transform 2d key points
        for idx in range(self.seq_len):
            kp_2d[idx, :, :2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx, :, :2],
                center_x=bbox[idx, 0],
                center_y=bbox[idx, 1],
                width=bbox[idx, 2],
                height=bbox[idx, 3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )

            kp_2d[idx, :, :2] = normalize_2d_kp(kp_2d[idx, :, :2], 224)

            kp_2d_tensor[idx] = kp_2d[idx]
            # feature[idx] = kp_2d[idx, :, :2]
            kp_3d_tensor[idx] = kp_3d[idx]
            # theta shape (85,)
            theta = np.concatenate((np.array([1., 0., 0.]), pose[idx], shape[idx]), axis=0)
            theta_tensor[idx] = theta

        target = {
            'features': feature,    # (seq_len, 49, 2)
            'theta': torch.from_numpy(theta_tensor).float(),  # (seq_len, 85)
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(),  # (seq_len, 49, 3)
            'kp_3d': torch.from_numpy(kp_3d_tensor).float(),  # (seq_len, test14, 3)
            'w_smpl': w_smpl,  # (seq_len,)
            'w_3d': w_3d,  # (seq_len,)
        }

        return target
