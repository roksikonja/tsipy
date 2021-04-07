from typing import Tuple

import numpy as np


def is_sorted(x: np.ndarray) -> bool:
    """Check if array is sorted.

    Args:
        x:  1-D array.

    Returns:
        True if array is sorted and False otherwise.
    """
    assert len(x.shape) == 1
    return all(x[i] <= x[i + 1] for i in range(len(x) - 1))


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

                k = k + 1

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
    """
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
