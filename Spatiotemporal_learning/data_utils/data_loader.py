import torch
from torch.utils.data import Dataset, IterableDataset
import gc
import numpy as np
import h5py
import os


class ItemDataset(Dataset):
    def __init__(self, args, feature_list, data_type='train'):
        super().__init__()
        self.args = args
        self.sequence_data = None
        self.feature_list = feature_list
        self.total_feature = len(feature_list)
        self.data_type = data_type

    def load_sequence_data(self, seq_data):
        self.sequence_data = h5py.File(seq_data, 'r')
        # os.posix_fadvise(os.fileno(seq_file), 0, self.sequence_data.id.get_filesize(), os.POSIX_FADV_DONTNEED)
        self.length = self.sequence_data.get(self.data_type + str(0)).shape[0]

    def get_feature_shapes(self):
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = torch.as_tensor(self.sequence_data[self.data_type + '0'][idx])
        y = torch.as_tensor(self.sequence_data[self.data_type + '1'][idx])
                
        if self.data_type == "predict":
            return X, y, idx
        else:
            return X, y


class PostProcessDataset(Dataset):
    def __init__(self, args, data_type='y_target'):
        super().__init__()
        self.args = args
        self.sequence_data = None
        self.data_type = data_type

    def load_sequence_data(self, seq_data):
        self.sequence_data = h5py.File(seq_data, 'r')
        self.length = self.sequence_data.get(self.data_type).shape[0]

        return self.sequence_data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_f = self.sequence_data[self.data_type][idx]
        return x_f

