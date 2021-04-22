import datetime
import os
from typing import Any, List, Tuple, Union

import numpy as np

__all__ = [
    "make_dir",
    "create_dir",
    "is_integer",
    "downsample_signal",
    "downsampling_indices_by_max_points",
    "transform_time_to_unit",
    "get_time_output",
    "is_sorted",
    "sort_inputs",
    "normalize",
    "denormalize",
    "find_nearest",
    "find_nearest_indices",
    "nonclipped_indices",
    "closest_binary_search",
    "get_window_indices",
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


def is_sorted(array: np.ndarray) -> bool:
    """Check if array is sorted.

    Args:
        array:  1-D array.

    Returns:
        True if array is sorted and False otherwise.
    """

    assert len(array.shape) == 1
    return all(array[i] <= array[i + 1] for i in range(len(array) - 1))


def sort_inputs(
    x: np.ndarray, y: np.ndarray, sort_axis: int
) -> Tuple[np.ndarray, np.ndarray]:

    if len(x.shape) != 2:
        raise ValueError("Array x with shape {} is not 2D.".format(x.shape))
    if len(y.shape) != 2:
        raise ValueError("Array y with shape {} is not 2D.".format(y.shape))

    x_indices_sorted = np.argsort(x[:, sort_axis])

    x = x[x_indices_sorted, :]
    y = y[x_indices_sorted, :]
    return x, y


def normalize(x: np.ndarray, mean: float, scale: float) -> np.ndarray:
    """Normalize array in the first dimension as given by y = (x - mean) / scale."""

    if len(x.shape) <= 1:
        y = (x - mean) / scale
    else:
        y = x
        y[:, 0] = (x[:, 0] - mean) / scale
    return y


def denormalize(y: np.ndarray, mean: float, scale: float) -> np.ndarray:
    """Denormalize array in the first dimension as given by x = y * scale + mean."""

    if len(y.shape) <= 1:
        x = scale * y + mean
    else:
        x = y
        x[:, 0] = scale * y[:, 0] + mean
    return x


def find_nearest(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = np.zeros_like(targets, dtype=np.int)

    for index, target in enumerate(targets):
        indices[index] = np.abs(values - target).argmin()

    return indices


def find_nearest_indices(
    sorted_values: np.ndarray, sorted_targets: np.ndarray
) -> np.ndarray:
    indices = []

    k = 0
    tar = sorted_targets[k]
    for i, val_i in enumerate(sorted_values):
        if tar < val_i:
            while tar < val_i:
                dist_i = abs(val_i - tar)
                dist_j = abs(sorted_values[max(i - 1, 0)] - tar)

                nearest_idx = i if dist_i <= dist_j else i - 1
                indices.append(nearest_idx)

                k += 1

                if k < len(sorted_targets):
                    tar = sorted_targets[k]
                else:
                    break

            if k == len(sorted_targets):
                break

    indices = np.array(indices)
    return indices


def nonclipped_indices(x: np.ndarray, n_scale: float = 5.0) -> np.ndarray:
    """Return non-clipped indices that are close to array mean.

    Non-clipped index i satisfies:
        x_mean + n_std * x_std >= x[i] >= x_mean - n_std * x_std.
    """

    clip_mean, clip_std = np.mean(x), np.std(x)

    lower_ids = np.greater_equal(x, clip_mean - n_scale * clip_std)
    upper_ids = np.less_equal(x, clip_mean + n_scale * clip_std)

    nonclipped_ids = np.logical_and(lower_ids, upper_ids)
    return nonclipped_ids


def closest_binary_search(array: np.ndarray, value: float) -> int:
    """Finds and returns the index of the closest element to ``value``.

    Args:
        array: Sorted 1D array.

        value: Value to be searched.
    """
    # pylint: disable=R1705
    left, right = 0, len(array) - 1
    best_id = left

    if value == float("inf"):
        return right
    elif value == -float("inf"):
        return left

    while left <= right:
        middle = left + (right - left) // 2
        if array[middle] < value:
            left = middle + 1
        elif array[middle] > value:
            right = middle - 1
        else:
            best_id = middle
            break

        if abs(array[middle] - value) < abs(array[best_id] - value):
            best_id = middle

    return best_id


def get_window_indices(x: np.ndarray, x_start: float, x_end: float) -> Tuple[int, int]:
    """Obtain the start and end indices in x that are in window [x_start, x_end].

    Args:
        x: Sorted 1-D array.
        x_start: Window start.
        x_end: Window end.

    Returns: A tuple of a start and end index of x, such that
        x_start <= x[x_start_id:x_end_id + 1] <= x_end.
    """

    x_start_id = closest_binary_search(array=x, value=x_start)
    # Handle a range of equal values
    if x_start_id != 0:
        while x[x_start_id - 1] == x_start:
            x_start_id -= 1

            if x_start_id == 0:
                break

    x_end_id = closest_binary_search(array=x, value=x_end)
    # Handle a range of equal values
    if x_end_id != (x.size - 1):
        while x[x_end_id + 1] == x_end:
            x_end_id += 1

            if x_end_id == (x.size - 1):
                break

    return x_start_id, x_end_id
