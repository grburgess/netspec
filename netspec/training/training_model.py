import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl


def mean_FE(output, target):
    return torch.mean(torch.abs(target - output) / target)


def max_FE(output, target):
    return torch.max(torch.abs(target - output) / target)


class TrainingNeuralNet(pl.LightningModule):
    def __init__(
        self,
        n_parameters: int,
        n_energies: int,
        n_hidden_layers: int,
        n_nodes: int,
        learning_rate: float = 1e-3,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate

        layers: List[nn.Module] = []

        current_input_dim = n_parameters

        for i in range(n_hidden_layers):

            layers.append(nn.Linear(current_input_dim, n_nodes))
            layers.append(nn.ReLU())

            if use_batch_norm:

                layers.append(nn.BatchNorm1d(n_nodes))

            current_input_dim = n_nodes

        layers.append(nn.Linear(current_input_dim, n_energies))

        # current_input_dim = n_input
        # for n in n_nodes:
        #     layers.append(nn.Linear(current_input_dim, n))

        #     layers.append(nn.ReLU())

        #     # if n < len(n_nodes) - 1:
        #     #
        #     current_input_dim = n

        # layers.append(nn.Linear(current_input_dim, n_output))

        self.layers: nn.Module = nn.Sequential(*layers)
        self.accuracy_max = max_FE
        self.accuracy_mean = mean_FE
        self.loss = nn.L1Loss(reduction="sum")

    def forward(self, x):
        return self.layers.forward(x)

    def training_step(self, batch, batch_idx: int) -> Dict[str, Any]:

        # get the data
        x, y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        nz = torch.nonzero(y)

        acc_max = self.accuracy_max(y_hat[nz], y[nz])
        acc_mean = self.accuracy_mean(y_hat[nz], y[nz])

        # Logging to TensorBoard by default

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        self.log(
            "performance",
            {"train_accuracy_max": acc_max, "train_accuracy_mean": acc_mean},
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        nz = y > 0

        acc_max = self.accuracy_max(y_hat[nz], y[nz])
        acc_mean = self.accuracy_mean(y_hat[nz], y[nz])

        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "performance",
            {
                "validation_accuracy_max": acc_max,
                "validation_accuracy_mean": acc_mean,
            },
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self) -> optim.NAdam:
        return optim.NAdam(self.parameters(), lr=self.learning_rate)


