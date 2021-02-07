"""
Functions and classes to build a trainable Dataset

"""

import torch
from torch.utils.data import Dataset, DataLoader
import random


class trainDataset(Dataset):
    '''

    Convenient class to support batch generation in the training phase.
    Inherits Torch.utils.data.Dataset

    '''

    def __init__(self, data, train_size=0.7):
        super(Dataset, self).__init__()
        self.data = data
        self.data_size = len(data)
        self._target_df = None
        self._target_size = 0

        # calculate the split size
        self.train_size = int(self.data_size * train_size)
        self.val_size = int(self.data_size * (1 - train_size))

        # split the data
        self.train_set, self.val_set = self.split_data(
            train_size=self.train_size)
        self.lookup = {
            'train': (self.train_set, self.train_size),
            'val': (self.val_set, self.val_size)
        }

        # Set training subset as target when initializing the Dataset
        self.set_split()

    def split_data(self, train_size):
        """
        Called only once at the beginning, given the data list returns three splitted sets
        for the three phases: training and validation

        """
        data = self.data
        print("Shuffling the data ... wait a minute")
        random.shuffle(data)
        train_set = data[:train_size]
        train_set = torch.tensor(
            train_set, dtype=torch.long)

        val_set = data[train_size:]
        val_set = torch.tensor(val_set, dtype=torch.long)

        return train_set, val_set

    def set_split(self, split="train"):
        """
        Switch between subsets

        """
        self._target_split = split
        self._target_df, self._target_size = self.lookup[split]

    def __getitem__(self, index):
        """
        From the target spli-set create a generator for batches

        """

        # index = the current pair (center_w, context_w) to analyze, 0 = take the center_w
        x = self._target_df[index, 0]
        # index = the current pair (center_w, context_w) to analyze, 1 = take the context_w
        y = self._target_df[index, 1]
        return x, y  # this will be used by the generator later on

    def __len__(self):
        return self._target_size


def make_batch(dataset, batch_size, shuffle, drop_last, device):
    """
    PyTorch DataLoader 

    """
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        drop_last=drop_last,
                        shuffle=shuffle
                        )

    for x, y in loader:
        yield x.to(device), y.to(device),
