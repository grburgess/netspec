import collections
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import astropy.units as u
import h5py
import numpy as np
from astromodels.core.parameter import Parameter
from astromodels.functions import Function1D, FunctionMeta
from astromodels.functions.function import Function1D, FunctionMeta
from astromodels.utils import get_user_data_path
from numba.core import descriptors
from ronswanson.database import dataclass
from torch import Tensor, dropout, from_numpy, nn, no_grad

from .training import Transformer
from .training.training_model import Layers
from .utils import (
    recursively_load_dict_contents_from_group,
    recursively_save_dict_contents_to_group,
)
from .utils.logging import setup_logger

log = setup_logger(__name__)


class MissingDataFile(RuntimeError):
    pass


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


# def save_model(model_name: str, transformer:, checkpoint_file: str) -> None:

#         pass


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
        transformer: Transformer,
        state_dict: Dict[Any, Any],
    ):

        self._neural_net: NeuralNet = NeuralNet(**asdict(model_params))
        self._transformer: Transformer = transformer

        self._neural_net.load_state_dict(state_dict)
        self._neural_net.eval()

        self._model_params = model_params

    @property
    def transformer(self) -> Transformer:
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

            transformer: Transformer = Transformer.from_file(f["transformer"])

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


class EmulatorModel(Function1D, metaclass=FunctionMeta):

    r"""
    description :
        A template model
    latex :
        $n.a.$
    parameters :
        K :
            desc : Normalization (freeze this to 1 if the template provides the normalization by itself)
            initial value : 1.0
        scale :
            desc : Scale for the independent variable. The templates are handled as if they contains the fluxes
                   at E = scale * x.This is useful for example when the template describe energies in the rest
                   frame, at which point the scale describe the transformation between rest frame energy and
                   observer frame energy. Fix this to 1 to neutralize its effect.
            initial value : 1.0
            min : 1e-5

        redshift:
            desc: redshift the energies
            initial value: 0.
            min: 0
            fix: True

    """

    def _custom_init_(
        self,
        model_name: str,
        other_name: Optional[str] = None,
        log_interp: bool = True,
    ):
        """
        Custom initialization for this model

        :param model_name: the name of the model, corresponding to the root of the .h5 file in the data directory
        :param other_name: (optional) the name to be used as name of the model when used in astromodels. If None
        (default), use the same name as model_name
        :return: none
        """

        self._log_interp: bool = log_interp

        # Get the data directory

        data_dir_path: Path = get_user_data_path()

        # Sanitize the data file

        filename_sanitized = data_dir_path.absolute() / f"{model_name}.h5"

        if not filename_sanitized.exists():

            log.error(f"The data file {filename_sanitized} does not exists.")

            raise MissingDataFile(
                f"The data file {filename_sanitized} does not exists."
            )

        # Open the template definition and read from it

        self._data_file: Path = filename_sanitized

        # use the file shadow to read

        self._model_storage: ModelStorage = ModelStorage.from_file(
            filename_sanitized
        )

        self._energies = self._model_storage.energies

        function_definition = collections.OrderedDict()

        description = "blah"

        function_definition["description"] = description

        function_definition["latex"] = "n.a."

        # Now get the metadata

        # description = template_file.description
        # name = template_file.name

        # Now build the parameters according to the content of the parameter grid

        parameters = collections.OrderedDict()

        parameters["K"] = Parameter("K", 1.0)
        parameters["scale"] = Parameter("scale", 1.0)
        parameters["redshift"] = Parameter("redshift", 0.0, free=False)

        for i, parameter_name in enumerate(
            self._model_storage.transformer.parameter_names
        ):

            par_range = np.array(
                [
                    self._model_storage.transformer.param_min[i],
                    self._model_storage.transformer.param_max[i],
                ]
            )

            parameters[parameter_name] = Parameter(
                parameter_name,
                np.median(par_range),
                min_value=par_range.min(),
                max_value=par_range.max(),
            )

        if other_name is None:

            super(EmulatorModel, self).__init__(
                model_name, function_definition, parameters
            )

        else:

            super(EmulatorModel, self).__init__(
                other_name, function_definition, parameters
            )

        # Finally prepare the interpolators

    def _set_units(self, x_unit, y_unit):

        self.K.unit = y_unit

        self.scale.unit = 1 / x_unit
        self.redshift.unit = u.dimensionless_unscaled

    # This function will be substituted during construction by another version with
    # all the parameters of this template

    def evaluate(self, x, K, scale, redshift, *args):

        net_output = self._model_storage.evaluate(
            np.array(args, dtype=np.float32)
        )

        if isinstance(x, u.Quantity):

            # Templates are always saved with energy in keV. We need to transform it to
            # a dimensionless quantity (actually we take the .value property) because otherwise
            # the logarithm below will fail.

            energies = np.array(
                x.to("keV").value, ndmin=1, copy=False, dtype=float
            )

            # Same for the scale

            scale = scale.to(1 / u.keV).value

        else:

            energies = x

        e_tilde = self._energies * scale

        if self._log_interp:

            return (
                np.interp(
                    np.log(energies * (1 + redshift)),
                    np.log(e_tilde),
                    net_output,
                )
                / scale
            )

        else:

            return (
                np.interp(energies * (1 + redshift), e_tilde, net_output)
                / scale
            )
