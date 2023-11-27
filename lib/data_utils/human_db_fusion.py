import joblib
import numpy as np
import os.path as osp
import sys
sys.path.append('.')

# from lib.core.config import VIBE_DB_DIR
path = '/data/wupeng/myProject/Now/BiVF_HMR/data/blvf_hmr_db'

dataset = {
    'vid_name': [],
    'frame_id': [],
    'joints3D': [],
    'joints2D': [],
    'bbox': [],
    'img_name': [],
    'features': [],
}

db1 = joblib.load(osp.join(path, 'h36m_train1_db.pt'))
db2 = joblib.load(osp.join(path, 'h36m_train2_db.pt'))
db3 = joblib.load(osp.join(path, 'h36m_train3_db.pt'))

for k in db1.keys():
    dataset[k].append(db1[k])
    dataset[k].append(db2[k])
    dataset[k].append(db3[k])
    dataset[k] = np.concatenate(dataset[k])

joblib.dump(dataset, osp.join(path, 'h36m_train_db.pt'))
