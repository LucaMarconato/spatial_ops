import random
from typing import List

import torch.utils.data

from spatial_ops.data import JacksonFischerDataset as jfd, Plate


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, shuffle=True):
        self.plates: List[Plate] = []
        for patient in jfd.patients:
            self.plates.extend(patient.plates)
        if shuffle:
            random.shuffle(self.plates)

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, item):
        ome = self.plates[item].get_ome()
        masks = self.plates[item].get_masks()
        masks_count = masks.max()
        data = (ome, masks, masks_count)
        return data


dataset = AutoEncoderDataset()
print(dataset[2])
