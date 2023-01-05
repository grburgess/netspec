import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError


class Layers(nn.Module):
    def __init__(
        self,
        n_parameters: int,
        n_energies: int,
        n_hidden_layers: int,
        n_nodes: int,
        dropout: Optional[float] = None,
        use_batch_norm: bool = False,
    ) -> None:

        super().__init__()

        self.relu = nn.ReLU()

        layers: List[nn.Module] = []

        current_input_dim = n_parameters

        for i in range(n_hidden_layers):

            layers.append(nn.Linear(current_input_dim, n_nodes))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))

            layers.append(self.relu)

            if use_batch_norm:

                layers.append(nn.BatchNorm1d(n_nodes))

            current_input_dim = n_nodes

        layers.append(nn.Linear(current_input_dim, n_energies))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers.forward(x)


class TrainingNeuralNet(pl.LightningModule):
    def __init__(
        self,
        n_parameters: int,
        n_energies: int,
        n_hidden_layers: int,
        n_nodes: int,
        use_batch_norm: bool = False,
        dropout: Optional[float] = None,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.learning_rate = learning_rate

        self.layers: nn.Module = Layers(
            n_parameters,
            n_energies,
            n_hidden_layers,
            n_nodes,
            dropout,
            use_batch_norm,
        )

        self.train_loss = MeanAbsoluteError()
        self.val_loss = MeanAbsoluteError()

        self.train_accuracy = MeanAbsolutePercentageError()

        self.val_accuracy = MeanAbsolutePercentageError()

    def forward(self, x):
        # return self.relu(self.layers.forward(x))

        return self.layers.forward(x)


    def initialize_weights(self) -> None:

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)


    def training_step(self, batch, batch_idx: int) -> Dict[str, Any]:

        # get the data
        x, y = batch

        pred = self.forward(x)

        loss = self.train_loss(pred, y)

        self.log(
            "train_loss",
            self.train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        # self.log(
        #     "performance",
        #     {"train_accuracy_max": acc_max, "train_accuracy_mean": acc_mean},
        #     on_step=False,
        #     on_epoch=True,
        # )

        return loss

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        pred = self.forward(x)

        loss = self.val_loss(pred, y)

        self.val_accuracy(pred, y)

        self.log(
            "val_loss",
            self.val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:

        #optimizer = optim.NAdam(self.parameters(), lr=self.learning_rate)
        optimizer = optim.NAdam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-4,
        #     max_lr=1e-1,
        #     step_size_up=5,
        #     mode="exp_range",
        #     gamma=0.85,
        # )

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=5, gamma=0.5
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "train_loss",
            # },
        }
