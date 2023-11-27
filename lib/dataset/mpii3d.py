# -*- coding: utf-8 -*-

from lib.dataset import MixDataset, ValidDataset


class Mpii3dTrain(MixDataset):
    def __init__(self, dataset_name, dataset_weight, seq_len, overlap):
        super(Mpii3dTrain, self).__init__(
            dataset_name=dataset_name,
            dataset_weight=dataset_weight,
            seq_len=seq_len,
            overlap=overlap
        )
        print(f'{dataset_name} - number of dataset objects {self.__len__()}')


class Mpii3dVal(ValidDataset):
    def __init__(self, dataset_name, seq_len, overlap):
        super(Mpii3dVal, self).__init__(
            dataset_name=dataset_name,
            seq_len=seq_len,
            overlap=overlap
        )
        print(f'{dataset_name} - number of dataset objects {self.__len__()}')
