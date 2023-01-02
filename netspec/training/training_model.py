import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl


def mean_FE(output, target, z_buffer=0.0):
    return torch.mean(
        (torch.abs((target+z_buffer) - output)) / (target + z_buffer)
    )


def max_FE(output, target, z_buffer=0.0):
    return torch.max(
        (torch.abs((target+z_buffer) - output) ) / (target + z_buffer)
    )


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

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        layers: List[nn.Module] = []

        current_input_dim = n_parameters

        for i in range(n_hidden_layers):

            layers.append(nn.Linear(current_input_dim, n_nodes))
            layers.append(self.relu)

            if use_batch_norm:

                layers.append(nn.BatchNorm1d(n_nodes))

            current_input_dim = n_nodes

        layers.append(nn.Linear(current_input_dim, n_energies))

        self.layers: nn.Module = nn.Sequential(*layers)
        self.accuracy_max = max_FE
        self.accuracy_mean = mean_FE
        self.loss = nn.L1Loss(reduction="sum")

    def forward(self, x):
        #return self.relu(self.layers.forward(x))

        return self.layers.forward(x)

    def training_step(self, batch, batch_idx: int) -> Dict[str, Any]:

        # get the data
        x, y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        nz = y > 0

        acc_max = self.accuracy_max(y_hat[nz], y[nz])
        acc_mean = self.accuracy_mean(y_hat[nz], y[nz])

        # acc_max = self.accuracy_max(y_hat, y, 1e-30)
        # acc_mean = self.accuracy_mean(y_hat, y, 1e-30)

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

        # acc_max = self.accuracy_max(y_hat, y, 1e-30)
        # acc_mean = self.accuracy_mean(y_hat, y, 1e-30)

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

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = optim.NAdam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-4,
        #     max_lr=1e-1,
        #     step_size_up=5,
        #     mode="exp_range",
        #     gamma=0.85,
        # )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
