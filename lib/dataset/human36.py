# -*- coding: utf-8 -*-

from lib.dataset import MixDataset, ValidDataset


class H36mTrain(MixDataset):
    def __init__(self, dataset_name, dataset_weight, seq_len, overlap):
        super(H36mTrain, self).__init__(
            dataset_name=dataset_name,
            dataset_weight=dataset_weight,
            seq_len=seq_len,
            overlap=overlap
        )
        print(f'{dataset_name} - number of dataset objects {self.__len__()}')


class H36mVal(ValidDataset):
    def __init__(self, dataset_name, seq_len, overlap):
        super(H36mVal, self).__init__(
            dataset_name=dataset_name,
            seq_len=seq_len,
            overlap=overlap
        )
        print(f'{dataset_name} - number of dataset objects {self.__len__()}')
