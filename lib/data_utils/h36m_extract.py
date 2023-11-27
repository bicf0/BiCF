#!/usr/bin/env python3

from os import path, makedirs
import tarfile
from tqdm import tqdm

subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']


# https://stackoverflow.com/a/6718435
def commonprefix(m):
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def extract_tgz(tgz_file, dest):
    if path.exists(dest):
        return
    with tarfile.open(tgz_file, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.isreg()]
        member_dirs = [path.dirname(m.name).split(path.sep) for m in members]
        base_path = path.sep.join(commonprefix(member_dirs))
        for m in members:
            m.name = path.relpath(m.name, base_path)
        tar.extractall(dest)


def extract_all():
    for subject_id in tqdm(subjects, ascii=True):
        out_dir = path.join('../../data/h36m/', subject_id)
        makedirs(out_dir, exist_ok=True)
        extract_tgz('../../data/h36m/Poses_D2_Positions_{}.tgz'.format(subject_id),
                    path.join(out_dir, 'MyPoseFeatures', 'D2_Positions'))
        extract_tgz('../../data/h36m/Poses_D3_Positions_mono_{}.tgz'.format(subject_id),
                    path.join(out_dir, 'MyPoseFeatures', 'D3_Positions_mono')),
        extract_tgz('../../data/h36m/Videos_{}.tgz'.format(subject_id),
                    path.join(out_dir, 'Videos'))
        extract_tgz('../../data/h36m/Segments_mat_gt_bs_{}.tgz'.format(subject_id),
                    path.join(out_dir, 'MySegmentsMat', 'ground_truth_bb'))


if __name__ == '__main__':
    extract_all()
