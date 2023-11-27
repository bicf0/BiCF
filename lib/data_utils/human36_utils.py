import os
import sys
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
import pathlib
import cdflib

sys.path.append('.')

from lib.models import spin
from lib.core.config import VIBE_DB_DIR
from lib.utils.utils import tqdm_enumerate
from lib.data_utils.kp_utils import convert_kps
from lib.data_utils.img_utils import get_bbox_from_kp2d
from lib.data_utils.feature_extractor import extract_features


# dataset_path=data/h36m
def read_train_data(dataset_path, debug=False, extract_img=False, protocol=1):
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    # user_list = [1, 5, 6, 7, 8]
    user_list = [8]
    imgs_path = os.path.join(dataset_path, 'images')
    pathlib.Path(imgs_path).mkdir(exist_ok=True)

    for user_i in user_list:
        user_name = 'S%d' % user_i
        print(user_name, "Start")
        # path with GT bounding boxes "missed bbox files, need generate bbox as mppi3d do
        # bbox_path = os.path.join(dataset_path, user_name, 'MySegmentMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')
        # go over all the sequences of each user

        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_i in tqdm(seq_list):
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

            # 3D pose file
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]

            # bbox file
            # bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            # bbox_h5py = h5py.File(bbox_file)

            vid_used_frames = []
            vid_used_joints = []
            vid_used_bbox = []
            # vid_segments = []
            vid_uniq_id = "subj" + str(user_i) + '_seq' + str(seq_i)

            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                # if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                cap = cv2.VideoCapture(vid_file)
                success, image = cap.read()
                if not success:
                    print("failed to read video {}".format(seq_name))
                    break

                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    image_name = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i + 1)
                    # img_out = ''
                    # save image
                    # if extract_img:
                    img_out = os.path.join(imgs_path, image_name)
                    cv2.imwrite(img_out, image)

                    # save 2d joint
                    joints_2d_raw = np.reshape(poses_2d[frame_i, :], [1, -1, 2])
                    joints_2d_raw = np.append(joints_2d_raw, np.ones((1, joints_2d_raw.shape[1], 1)), axis=2)
                    joints_2d = convert_kps(joints_2d_raw, "h36m", "spin").reshape((-1, 3))

                    # save 3d point
                    joints_3d_raw = np.reshape(poses_3d[frame_i, :], [1, -1, 3]) / 1000.
                    joints_3d = convert_kps(joints_3d_raw, "h36m", "spin").reshape(-1, 3)

                    # generate bbox and save bbox
                    """if bbox is available"""
                    # mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]].value.T
                    # ys, xs = np.where(mask == 1)
                    # bbox = np.array([np.min(xs), np.min(ys), np.max(xs) + 1, np.max(ys) + 1])
                    # # center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                    # b_w, b_h = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    # bbox = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, b_w, b_h])
                    """if bbox is not available, then generate using 2d joints"""
                    bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

                    joints_3d = joints_3d - joints_3d[39]  # 4 is the root

                    dataset['vid_name'].append(vid_uniq_id)
                    dataset['frame_id'].append(image_name.split(".")[0])
                    dataset['img_name'].append(img_out)
                    dataset['joints2D'].append(joints_2d)
                    dataset['joints3D'].append(joints_3d)
                    dataset['bbox'].append(bbox)
                    vid_used_frames.append(img_out)
                    vid_used_joints.append(joints_2d)
                    vid_used_bbox.append(bbox)

            # generate features
            features = extract_features(model, np.array(vid_used_frames),
                                        vid_used_bbox,
                                        kp_2d=np.array(vid_used_joints),
                                        dataset='spin', debug=False)
            dataset['features'].append(features)
        print(user_name, "End")

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset


def read_test_data(dataset_path, debug=False, extract_img=False, protocol=1):
    dataset = {
        'vid_name': [],
        'frame_id': [],
        'joints3D': [],
        'joints2D': [],
        'bbox': [],
        'img_name': [],
        'features': [],
    }

    model = spin.get_pretrained_hmr()

    user_list = [9, 11]

    imgs_path = os.path.join(dataset_path, 'images')
    pathlib.Path(imgs_path).mkdir(exist_ok=True)

    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes "missed bbox files, need generate bbox as mppi3d do
        # bbox_path = os.path.join(dataset_path, user_name, 'MySegmentMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')
        # go over all the sequences of each user

        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_i in seq_list:
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

            # 3D pose file
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]

            # bbox file
            # bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            # bbox_h5py = h5py.File(bbox_file)

            vid_used_frames = []
            vid_used_joints = []
            vid_used_bbox = []
            # vid_segments = []
            vid_uniq_id = "subj" + str(user_i) + '_seq' + str(seq_i)

            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                # if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                cap = cv2.VideoCapture(vid_file)
                success, image = cap.read()
                if not success:
                    print("failed to read video {}".format(seq_name))
                    break

                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    image_name = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i + 1)
                    # img_out = ''
                    # save image
                    # if extract_img:
                    img_out = os.path.join(imgs_path, image_name)
                    cv2.imwrite(img_out, image)
                        # print(img_out)
                    # print(img_out)
                    # save 2d joint
                    joints_2d_raw = np.reshape(poses_2d[frame_i, :], [1, -1, 2])
                    joints_2d_raw = np.append(joints_2d_raw, np.ones((1, joints_2d_raw.shape[1], 1)), axis=2)
                    joints_2d = convert_kps(joints_2d_raw, "h36m", "spin").reshape((-1, 3))

                    # save 3d point
                    joints_3d_raw = np.reshape(poses_3d[frame_i, :], [1, -1, 3]) / 1000.
                    joints_3d = convert_kps(joints_3d_raw, "h36m", "spin").reshape(-1, 3)

                    # generate bbox and save bbox
                    """if bbox is available"""
                    # mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]].value.T
                    # ys, xs = np.where(mask == 1)
                    # bbox = np.array([np.min(xs), np.min(ys), np.max(xs) + 1, np.max(ys) + 1])
                    # center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
                    # b_w, b_h = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    # bbox = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, b_w, b_h])
                    """if bbox is not available, then generate using 2d joints"""
                    bbox = get_bbox_from_kp2d(joints_2d[~np.all(joints_2d == 0, axis=1)]).reshape(4)

                    joints_3d = joints_3d - joints_3d[39]  # 4 is the root

                    dataset['vid_name'].append(vid_uniq_id)
                    dataset['frame_id'].append(image_name.split(".")[0])
                    dataset['img_name'].append(img_out)
                    dataset['joints2D'].append(joints_2d)
                    dataset['joints3D'].append(joints_3d)
                    dataset['bbox'].append(bbox)
                    vid_used_frames.append(img_out)
                    vid_used_joints.append(joints_2d)
                    vid_used_bbox.append(bbox)

            # generate features
            features = extract_features(model, np.array(vid_used_frames),
                                        vid_used_bbox,
                                        kp_2d=np.array(vid_used_joints),
                                        dataset='spin', debug=False)
            dataset['features'].append(features)

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
    dataset['features'] = np.concatenate(dataset['features'])

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='dataset directory', default='data/h36m')
    args = parser.parse_args()

    # dataset = read_test_data(args.dir)
    # joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'h36m_val_db.pt'))

    dataset = read_train_data(args.dir)
    joblib.dump(dataset, osp.join(VIBE_DB_DIR, 'h36m_train3_db.pt'))

    # python lib/data_utils/human36_utils.py --dir ./data/h36m
