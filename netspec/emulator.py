import collections

from pathlib import Path
from typing import Optional

import astropy.units as u

import numpy as np
from astromodels.core.parameter import Parameter
from astromodels.functions import Function1D, FunctionMeta
from astromodels.functions.function import Function1D, FunctionMeta
from astromodels.utils import get_user_data_path

from .utils.logging import setup_logger
from .utils.model_utils import ModelStorage

log = setup_logger(__name__)


class MissingDataFile(RuntimeError):
    pass


# def save_model(model_name: str, transformer:, checkpoint_file: str) -> None:

#         pass


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
