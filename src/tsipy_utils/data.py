import datetime
import os
from typing import Any, List, Union

import numpy as np

__all__ = [
    "make_dir",
    "create_dir",
    "is_integer",
    "downsample_signal",
    "downsampling_indices_by_max_points",
    "transform_time_to_unit",
    "get_time_output",
]


def make_dir(directory: str) -> str:
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def create_dir(results_dir_path: str, dir_name: str = "results") -> str:
    """Creates a directory with added timestamp of creation."""
    results_dir = make_dir(
        os.path.join(
            results_dir_path,
            datetime.datetime.now().strftime(f"%m-%d_%H-%M-%S_{dir_name}"),
        )
    )
    return results_dir


def is_integer(num: Any) -> bool:
    """Checks if the input has an integer type."""
    return isinstance(num, (int, np.int, np.int32, np.int64))


def downsample_signal(x: np.ndarray, k: int = 1) -> np.ndarray:
    """Downsamples a signal uniformly with a rate of ``k``."""
    if not is_integer(k):
        raise Exception("Downsampling factor must be an integer.")
    if k > 1:
        return x[::k]
    else:
        return x


def downsampling_indices_by_max_points(
    x: np.ndarray, max_points: int = 100_000
) -> np.ndarray:
    """Computes indices of a uniformly downsampled signal of length ``max_points``."""
    indices = np.ones_like(x, dtype=np.bool)
    if x.shape[0] > max_points:
        downsampling_factor = x.shape[0] // max_points

        indices = np.zeros_like(x, dtype=np.bool)
        indices[::downsampling_factor] = True

    return indices


def transform_time_to_unit(
    t: np.ndarray,
    t_label: str = "year",
    start: datetime.datetime = datetime.datetime(1996, 1, 1),
) -> np.ndarray:
    """Transforms time unit to `t_label` starting at `start`.

    Examples:
        >>> import numpy as np
        >>> t = np.arange(0, 366, 365.25 / 4)  # Time in days
        >>> transform_time_to_unit(t)  # Transformed to years
        array([1996.  , 1996.25, 1996.5 , 1996.75, 1997.  ])

    """
    if t_label == "year":
        t = np.array(
            [start.year + ((start.timetuple().tm_yday - 1) + day) / 365.25 for day in t]
        )
    else:
        raise ValueError("Invalid value for t_label {}.".format(t_label))

    return t


def get_time_output(
    t_nns: List[np.ndarray],
    n_per_unit: int,
    min_time: Union[np.ndarray, float] = None,
    max_time: Union[np.ndarray, float] = None,
) -> np.ndarray:
    """Creates a time array with n_per_unit` elements per unit."""
    if min_time is None:
        min_time = np.max([np.min(t_nn) for t_nn in t_nns])

    if max_time is None:
        max_time = np.min([np.max(t_nn) for t_nn in t_nns])

    n_out = int(n_per_unit * (max_time - min_time) + 1)

    t_out = np.linspace(min_time, max_time, n_out)
    return t_out
