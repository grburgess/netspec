from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from astromodels.utils import get_user_data_path
from ronswanson.database import dataclass
from torch import Tensor, from_numpy, nn, no_grad

import netspec.training.training_data_tools as tdt

from ..utils import (
    recursively_load_dict_contents_from_group,
    recursively_save_dict_contents_to_group,
)
from ..utils.logging import setup_logger

log = setup_logger(__name__)


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


class NeuralNet(nn.Module):
    def __init__(
        self,
        n_parameters: int,
        n_energies: int,
        n_hidden_layers: int,
        n_nodes: int,
        use_batch_norm: bool = False,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.layers: nn.Module = Layers(
            n_parameters,
            n_energies,
            n_hidden_layers,
            n_nodes,
            dropout,
            use_batch_norm,
        )

    def forward(self, x):
        return self.layers(x)


@dataclass
class ModelParams:
    n_parameters: int
    n_energies: int
    n_hidden_layers: int
    n_nodes: int
    use_batch_norm: bool
    dropout: Optional[float] = None


class ModelStorage:
    def __init__(
        self,
        model_params: ModelParams,
        transformer: tdt.Transformer,
        state_dict: Dict[Any, Any],
    ):

        self._neural_net: NeuralNet = NeuralNet(**asdict(model_params))
        self._transformer: tdt.Transformer = transformer

        self._neural_net.load_state_dict(state_dict)
        self._neural_net.eval()

        self._model_params = model_params

    @property
    def transformer(self) -> tdt.Transformer:
        return self._transformer

    def evaluate(self, params) -> np.ndarray:

        transformed_params = self._transformer.transform_parameters(params)

        with no_grad():

            output: Tensor = self._neural_net(from_numpy(transformed_params))

        return self._transformer.inverse_values(output.numpy())

    @property
    def energies(self) -> np.ndarray:
        return self._transformer.energies

    @classmethod
    def from_file(cls, file_name: str) -> "ModelStorage":

        with h5py.File(file_name, "r") as f:

            transformer: tdt.Transformer = tdt.Transformer.from_file(
                f["transformer"]
            )

            state_dict = recursively_load_dict_contents_from_group(
                f, "state_dict"
            )

            model_params = ModelParams(**f.attrs)

        return cls(
            model_params=model_params,
            transformer=transformer,
            state_dict=state_dict,
        )

    def to_file(self, file_name: str) -> None:

        with h5py.File(file_name, "w") as f:

            transform_group: h5py.Group = f.create_group("transformer")

            self._transformer.to_file(transform_group)

            recursively_save_dict_contents_to_group(
                f, "state_dict", self._neural_net.state_dict()
            )

            for k, v in asdict(self._model_params).items():

                f.attrs[k] = v

    def save_to_user_dir(
        self, model_name: str, overwrite: bool = False
    ) -> None:

        # Get the data directory

        data_dir_path: Path = get_user_data_path()

        # Sanitize the data file

        filename_sanitized = data_dir_path.absolute() / f"{model_name}.h5"

        if filename_sanitized.exists() and (not overwrite):

            log.error(f"{model_name}.h5 already exists!")

            raise RuntimeError(f"{model_name}.h5 already exists!")

        self.to_file(filename_sanitized.as_posix())
