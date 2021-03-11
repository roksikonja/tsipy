from numbers import Real

import numpy as np


def normalize(x, mean, std):
    if isinstance(x, Real) or len(x.shape) <= 1:
        y = (x - mean) / std
    else:
        y = x
        y[:, 0] = (x[:, 0] - mean) / std
    return y


def denormalize(y, mean, std):
    if isinstance(y, Real) or len(y.shape) <= 1:
        x = std * y + mean
    else:
        x = y
        x[:, 0] = std * y[:, 0] + mean
    return x


def find_nearest(values, targets):
    indices = np.zeros_like(targets, dtype=np.int)

    for index, target in enumerate(targets):
        indices[index] = np.abs(values - target).argmin()

    return indices


def find_nearest_indices(sorted_values, sorted_targets) -> np.ndarray:
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


def clipping_indices(x, n_std=5):
    """
    Return clip indices in x_mean - n_std * x_std >= x or x >= x_mean + n_std * x_std.
    """
    clip_mean, clip_std = np.mean(x), np.std(x)
    lower_indices = np.greater_equal(x, clip_mean - n_std * clip_std)
    upper_indices = np.less_equal(x, clip_mean + n_std * clip_std)

    clip_indices = np.logical_and(lower_indices, upper_indices)
    return clip_indices


def closest_binary_search(array, value):
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


def get_window_indices(x, x_start, x_end):
    """Obtain the start and end indices in x that are in window [x_start, x_end].

    Args:
        x: Sorted 1-D array.
        x_start: Window start.
        x_end: Window end.

    Returns: A tuple of a start and end index of x, such that
        x_start <= x[x_start_id:x_end_id + 1] <= x_end.
    """
    x_start_id = closest_binary_search(array=x, value=x_start)
    # handle a range of equal values
    if x_start_id != 0:
        while x[x_start_id - 1] == x_start:
            x_start_id = x_start_id - 1

            if x_start_id == 0:
                break

    x_end_id = closest_binary_search(array=x, value=x_end)
    # handle a range of equal values
    if x_end_id != (x.size - 1):
        while x[x_end_id + 1] == x_end:
            x_end_id = x_end_id + 1

            if x_end_id == (x.size - 1):
                break

    return x_start_id, x_end_id
