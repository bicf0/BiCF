# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.dataset import MixDataset, ValidDataset
# from lib.core.config_model_data import THREEDPW_DIR


class ThreedpwTrain(MixDataset):
    def __init__(self, dataset_name, dataset_weight, seq_len, overlap):
        super(ThreedpwTrain, self).__init__(
            dataset_name=dataset_name,
            dataset_weight=dataset_weight,
            seq_len=seq_len,
            overlap=overlap
        )
        print(f'{dataset_name} - number of dataset objects {self.__len__()}')


class ThreedpwVal(ValidDataset):
    def __init__(self, dataset_name, seq_len, overlap):
        super(ThreedpwVal, self).__init__(
            dataset_name=dataset_name,
            seq_len=seq_len,
            overlap=overlap
        )
        print(f'{dataset_name} - number of dataset objects {self.__len__()}')
