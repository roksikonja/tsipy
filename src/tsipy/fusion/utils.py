from typing import List, Tuple

import numpy as np


def build_labels(xs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    labels = []
    x_labels = []
    for i, x in enumerate(xs):
        label = i + 1
        x_label = np.ones_like(x, dtype=np.int) * label

        labels.append(label)
        x_labels.append(x_label)

    labels = np.array(labels)
    x_labels = np.hstack(x_labels)
    return labels, x_labels


def build_output_labels(x_out: np.ndarray) -> np.ndarray:
    return -np.ones_like(x_out, dtype=np.int)


def concatenate_labels(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = np.vstack((x, labels)).T
    return x
