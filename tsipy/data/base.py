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


def find_nearest_indices(sorted_values, sorted_targets):
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
