import numpy as np
from typing import Any, Dict, Optional, Tuple, List
from torch import optim, nn, utils, Tensor
import torch
import pytorch_lightning as pl
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
)


from netspec.emulator import ModelParams, ModelStorage, Layers
from netspec.training.training_data_tools import Transformer


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
        use_mape: bool = False,
    ) -> None:
        super().__init__()

        self._n_parameters: int = n_parameters
        self._n_energies: int = n_energies
        self._n_hidden_layers: int = n_hidden_layers
        self._n_nodes: int = n_nodes
        self._dropout: Optional[float] = dropout
        self._use_batch_norm: bool = use_batch_norm

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

        self._use_mape: bool = use_mape

        if not self._use_mape:

            self.train_accuracy = SymmetricMeanAbsolutePercentageError()

            self.val_accuracy = SymmetricMeanAbsolutePercentageError()

        else:

            self.train_accuracy = MeanAbsolutePercentageError()

            self.val_accuracy = MeanAbsolutePercentageError()

    def forward(self, x):

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

        return loss

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        pred = self.forward(x)

        loss = self.val_loss(pred, y)

        if self._use_mape:

            nz_idx = torch.nonzero(y)

            self.val_accuracy(pred[nz_idx], y[nz_idx])

        else:

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

        # optimizer = optim.NAdam(self.parameters(), lr=self.learning_rate)
        optimizer = optim.NAdam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=1e-4,
            max_lr=1e-1,
            step_size_up=5,
            mode="exp_range",
            gamma=0.85,
        )

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

    def save_model(
        self,
        file_name: Union[str, Path],
        checkpoint,
        transformer: Transformer,
        overwrite: bool = False,
    ) -> None:

        model_params: ModelParams = ModelParams(
            self._n_parameters,
            self._n_energies,
            self._n_hidden_layers,
            self._n_nodes,
            self._use_batch_norm,
            self._dropout,
        )

        model_storage: ModelStorage = ModelStorage(
            model_params, transformer, checkpoint["state_dict"]
        )

        model_storage.save_to_user_dir(file_name, overwrite)
