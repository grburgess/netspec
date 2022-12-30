from typing import Optional, Tuple, List
from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl

n_neurons = 256
len_parameters = 7
len_spectra = 50

activation = nn.Sigmoid


def meanFE(output, target):
    return torch.mean(torch.abs((target - output) / target))


def maxFE(output, target):
    return torch.max(torch.abs((target - output) / target))


class NeuralNet(pl.LightningModule):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_nodes: List[int],
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate

        layers: List[nn.Module] = []

        current_input_dim = n_input
        for n in n_nodes:
            layers.append(nn.Linear(current_input_dim, n))

            layers.append(nn.ReLU())

            # if n < len(n_nodes) - 1:
            #
            current_input_dim = n

        layers.append(nn.Linear(current_input_dim, n_output))

        self.layers: nn.Module = nn.Sequential(*layers)
        self.accuracy_max = maxFE
        self.accuracy_mean = meanFE
        self.loss = nn.L1Loss(reduction="sum")

    def forward(self, x):
        return self.layers.forward(x)

    def training_step(self, batch, batch_idx: int):

        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        acc_max = self.accuracy_max(y_hat, y)
        acc_mean = self.accuracy_mean(y_hat, y)

        self.log("acc_max", acc_max)
        self.log("acc_mean", acc_mean)
        self.log("hp_metric_max", acc_max, on_step=False, on_epoch=True)
        self.log("hp_metric_mean", acc_mean, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> optim.NAdam:
        return optim.NAdam(self.parameters(), lr=self.learning_rate)


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
