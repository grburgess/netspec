from typing import Union, Optional, List
import h5py
import numba as nb
import numpy as np
from ronswanson import Database


def first_last_nonzero(arr, axis, invalid_val=-1):
    mask = arr > 0.0

    first = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    mask = np.flip(mask, axis=1)

    last = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return first, -last


def min_max_fit(X):

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    return x_min, x_max


@nb.njit(cache=True)
def min_max_transform(X, x_min, min_max_difference) -> np.ndarray:

    x_std = (X - x_min) / min_max_difference

    return x_std


@nb.njit(cache=True)
def min_max_inverse(X_scaled, x_min, x_max) -> np.ndarray:

    return X_scaled * (x_max - x_min) + x_min


@nb.njit(cache=True)
def arcsinh(x):
    return np.arcsinh(x)


@nb.njit(cache=True)
def sinh(x):
    return np.sinh(x)


class Transformer:
    def __init__(
        self,
        param_min: np.ndarray,
        param_max: np.ndarray,
        value_min: np.ndarray,
        value_max: np.ndarray,
        energies: np.ndarray,
        parameter_names: Optional[List[str]] = None,
    ) -> None:

        self._param_min = param_min
        self._param_max = param_max

        # if the difference is zero, this will
        # blow up, so we fix it to one

        self._param_min_max_difference = param_max - param_min

        idx = self._param_min_max_difference == 0.0

        self._param_min_max_difference[idx] = 1

        self._value_min = value_min
        self._value_max = value_max

        self._value_min_max_difference = value_max - value_min

        idx = self._value_min_max_difference == 0.0

        self._value_min_max_difference[idx] = 1

        self._energies: np.ndarray = energies

        self._parameter_names: Optional[List[str]] = parameter_names

    @property
    def parameter_names(self) -> Optional[List[str]]:
        return self._parameter_names

    @property
    def energies(self) -> np.ndarray:
        return self._energies

    def to_file(self, file_name: Union[str, h5py.Group]) -> None:

        if isinstance(file_name, h5py.Group):

            f = file_name

            is_file: bool = False

        else:

            f = h5py.File(file_name, "w")
            is_file = True

        if self._parameter_names is not None:

            for i in range(len(self._parameter_names)):

                f.attrs[f"par_{i}"] = self._parameter_names[i]

        f.create_dataset("param_min", data=self._param_min, compression="gzip")
        f.create_dataset("param_max", data=self._param_max, compression="gzip")
        f.create_dataset("value_min", data=self._value_min, compression="gzip")
        f.create_dataset("value_max", data=self._value_max, compression="gzip")

        f.create_dataset("energies", data=self._energies, compression="gzip")

        if is_file:

            f.close()

    @classmethod
    def from_file(cls, file_name: Union[str, h5py.Group]) -> "Transformer":

        if isinstance(file_name, h5py.Group):

            f = file_name

            is_file: bool = False

        else:

            f = h5py.File(file_name, "r")
            is_file = True

        if "par_0" in f.attrs:

            parameter_names = []

            for i in range(len(f.attrs)):

                parameter_names.append(f.attrs[f"par_{i}"])

        else:

            parameter_names = None

        param_min: np.ndarray = f["param_min"][()]
        param_max: np.ndarray = f["param_max"][()]

        value_min: np.ndarray = f["value_min"][()]
        value_max: np.ndarray = f["value_max"][()]

        energies: np.ndarray = f["energies"][()]

        if is_file:

            f.close()

        return cls(
            param_min=param_min,
            param_max=param_max,
            value_min=value_min,
            value_max=value_max,
            energies=energies,
            parameter_names=parameter_names,
        )

    @property
    def param_min(self) -> np.ndarray:
        return self._param_min

    @property
    def param_max(self) -> np.ndarray:
        return self._param_max

    def transform_parameters(self, parameters: np.ndarray) -> np.ndarray:

        return min_max_transform(parameters, self._param_min, self._param_max)

    def inverse_parameters(
        self, transformed_parameters: np.ndarray
    ) -> np.ndarray:

        return min_max_inverse(
            transformed_parameters,
            self._param_min,
            self._param_min_max_difference,
        )

    def transform_values(self, values: np.ndarray) -> np.ndarray:

        return min_max_transform(
            arcsinh(values), self._value_min, self._value_min_max_difference
        )

    def inverse_values(self, transformed_values: np.ndarray) -> np.ndarray:

        return sinh(
            min_max_inverse(
                transformed_values, self._value_min, self._value_max
            )
        )


class TransformedData:
    def __init__(
        self,
        params: np.ndarray,
        values: np.ndarray,
        param_min: np.ndarray,
        param_max: np.ndarray,
        value_min: np.ndarray,
        value_max: np.ndarray,
        energies: np.ndarray,
    ) -> None:

        self._params = params
        self._values = values

        self._param_min = param_min
        self._param_max = param_max

        self._value_min = value_min
        self._value_max = value_max

        self._transformer: Transformer = Transformer(
            param_min, param_max, value_min, value_max, energies
        )

    def to_file(self, file_name: str) -> None:

        with h5py.File(file_name, "w") as f:

            f.create_dataset(
                "param_min", data=self._param_min, compression="gzip"
            )
            f.create_dataset(
                "param_max", data=self._param_max, compression="gzip"
            )
            f.create_dataset(
                "value_min", data=self._value_min, compression="gzip"
            )
            f.create_dataset(
                "value_max", data=self._value_max, compression="gzip"
            )

            f.create_dataset(
                "energies", data=self._transformer.energies, compression="gzip"
            )

            f.create_dataset("params", data=self._params, compression="gzip")
            f.create_dataset("values", data=self._values, compression="gzip")

    @classmethod
    def from_file(cls, file_name: str) -> "TransformedData":

        with h5py.File(file_name, "r") as f:

            param_min: np.ndarray = f["param_min"][()]
            param_max: np.ndarray = f["param_max"][()]

            value_min: np.ndarray = f["value_min"][()]
            value_max: np.ndarray = f["value_max"][()]

            params: np.ndarray = f["params"][()]
            values: np.ndarray = f["values"][()]
            energies: np.ndarray = f["energies"][()]

        return cls(
            params=params,
            values=values,
            param_min=param_min,
            param_max=param_max,
            value_min=value_min,
            value_max=value_max,
            energies=energies,
        )

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def params(self) -> np.ndarray:
        return self._params


def prepare_training_data(
    database: Database,
    file_name_stub: str,
    normalization_factor: float = 1.0,
    dirty_data_check: bool = False,
) -> None:

    # remove all zero rows
    zero_idx = database.values.sum(axis=1) == 0

    # if we still have dirty data

    if dirty_data_check:

        ok_idx = np.ones(database.n_entries, dtype=bool)

        first, last = first_last_nonzero(database.values, axis=1)

        for i, datum in enumerate(database.values):

            if np.any(datum[first[i] : last[i]] == 0):

                ok_idx[i] = 0

        zero_idx = zero_idx & ok_idx

    # scale the parmeters
    param_min, param_max = min_max_fit(database.grid_points[~zero_idx])

    # arcsinh the data
    # this is similar to a log transform
    # but it preserves zeros

    transformed_data = arcsinh(
        database.values[~zero_idx].astype("float64") * normalization_factor
    )

    # now min, max data

    value_min, value_max = min_max_fit(transformed_data)

    param_min = param_min
    param_max = param_max

    value_min = value_min
    value_max = value_max

    # now squash the it all

    transformer: Transformer = Transformer(
        param_min=param_min,
        param_max=param_max,
        value_min=value_min,
        value_max=value_max,
        energies=database.energy_grid,
        parameter_names=database.parameter_names,
    )

    squashed_data = transformer.transform_values(
        database.values[~zero_idx].astype("float64") * normalization_factor
    )

    squashed_params = transformer.transform_parameters(
        database.grid_points[~zero_idx]
    )

    transformed_data: TransformedData = TransformedData(
        params=squashed_params,
        values=squashed_data,
        param_min=param_min,
        param_max=param_max,
        value_min=value_min,
        value_max=value_max,
        energies=database.energy_grid,
    )

    transformed_data.to_file(f"{file_name_stub}.h5")
    transformer.to_file(f"{file_name_stub}_transformer.h5")
