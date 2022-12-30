from typing import List, Optional, Tuple

import torch
from torch import Tensor, utils


class PrepareData(utils.data.Dataset):
    def __init__(self, X, y, split_ratio: float = 0.2):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

        self._split_ratio: float = split_ratio

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Lore:
    """
    Controls the data
    """

    def __init__(self, x, y, split_ratio: float = 0.2) -> None:

        self._data_set: utils.data.DataSet = PrepareData(x, y)

        self._train_set_size: Optional[int] = None
        self._valid_set_size: Optional[int] = None

        self._train_set: Optional[utils.data.DataSet] = None
        self._valid_set: Optional[utils.data.DataSet] = None

        # split the data

        self._data_was_split: bool = False

        self.split_data(split_ratio)

    def split_data(self, split_ratio: float) -> None:

        # Random split
        self._train_set_size = int(len(self._data_set) * split_ratio)
        self._valid_set_size = len(self._data_set) - self._train_set_size

        self._train_set, self._valid_set = utils.data.random_split(
            self._data_set, [self._train_set_size, self._valid_set_size]
        )

        self._data_was_split = True

    def train_loader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 1
    ) -> utils.data.DataLoader:

        train_loader = utils.data.DataLoader(
            self._train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        return train_loader

    def valid_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 1
    ) -> utils.data.DataLoader:

        valid_loader = utils.data.DataLoader(
            self._valid_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        return valid_loader
