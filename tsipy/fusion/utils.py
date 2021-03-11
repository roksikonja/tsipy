import numpy as np


def build_sensor_labels(x):
    labels = []
    x_labels = []
    for i, signal in enumerate(x):
        label = i + 1
        x_label = np.ones_like(signal, dtype=np.int) * label

        labels.append(label)
        x_labels.append(x_label)

    labels = np.array(labels)
    x_labels = np.hstack(x_labels)
    return labels, x_labels


def build_output_labels(x_out):
    return -np.ones_like(x_out, dtype=np.int)


def concatenate_labels(x, labels, sort_axis=None):
    x = np.vstack((x, labels)).T

    if sort_axis is not None:
        x_indices_sorted = np.argsort(x[:, sort_axis])
        x = x[x_indices_sorted, :]

    return x
