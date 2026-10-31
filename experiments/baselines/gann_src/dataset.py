import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from utils import sorted_aphanumeric, sliding_window_corrcoef


class RtFCDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        label_dir=None,
        window=None,
        slice=None,
        normalize=False,
        transpose=False,
    ):
        super(RtFCDataset, self).__init__()
        self.normalize = normalize
        self.transpose = transpose
        self.data_path = []
        self.labels = []

        if data_dir is not None:
            self.data_path = [
                os.path.join(data_dir, name)
                for name in sorted_aphanumeric(os.listdir(data_dir))
            ]
            if slice:
                self.data_path = self.data_path[slice]

        self.n_data = len(self.data_path)
        self.window = [window] * self.n_data

        if isinstance(label_dir, str):
            self.labels = [
                torch.from_numpy(np.loadtxt(label_dir, dtype=int))
            ] * self.n_data
        elif label_dir is not None:
            self.labels = label_dir

        if self.n_data > 0:
            print('number of samples:', self.n_data)
            print('window:', window)
            print('data_path:', f'{self.data_path[0]},...,{self.data_path[-1]}')
            if isinstance(label_dir, str):
                print('label_dir:', label_dir)

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path = list(self.data_path) + list(other.data_path)
        self.labels = list(self.labels) + list(other.labels)
        self.window = list(self.window) + list(other.window)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        path = self.data_path[idx]
        load = np.loadtxt
        data = sliding_window_corrcoef(
            load(path, dtype=np.float32).T
            if self.transpose
            else load(path, dtype=np.float32),
            self.window[idx],
        )
        data = torch.from_numpy(data)

        if self.normalize:
            data += self.normalize * torch.eye(data.shape[-1])

        label = self.labels[idx]

        return data, label
