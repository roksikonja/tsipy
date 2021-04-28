from typing import Tuple

import numpy as np

from tsipy.correction import SignalGenerator
from tsipy.fusion.utils import (
    build_and_concat_label_mask,
    build_and_concat_label_mask_output,
)
from tsipy.utils import sort_inputs


def load_data_with_labels(random_seed: int) -> Tuple[np.ndarray, ...]:
    """Generates a dataset of two signals and constructs signal labels."""
    signal_generator = SignalGenerator(
        length=10_000, add_degradation=False, random_seed=random_seed
    )

    x_a, y_a = signal_generator["a"]
    x_b, y_b = signal_generator["b"]

    x_a = build_and_concat_label_mask(x_a, label=1)
    x_b = build_and_concat_label_mask(x_b, label=2)
    x_out = build_and_concat_label_mask_output(signal_generator.x)

    # Concatenate signals and sort by x[:, 0]
    x = np.vstack((x_a, x_b))
    y = np.reshape(np.hstack((y_a, y_b)), newshape=(-1, 1))
    x, y = sort_inputs(x, y, sort_axis=0)

    return x_a, x_b, x, y_a, y_b, y, x_out, signal_generator.x, signal_generator.y


def load_data_without_labels(random_seed: int) -> Tuple[np.ndarray, ...]:
    """Generates a dataset of two signals."""
    signal_generator = SignalGenerator(
        length=10_000, add_degradation=False, random_seed=random_seed
    )

    x_a, y_a = signal_generator["a"]
    x_b, y_b = signal_generator["b"]

    # Concatenate signals and sort by x[:, 0]
    x = np.reshape(np.hstack((x_a, x_b)), newshape=(-1, 1))
    y = np.reshape(np.hstack((y_a, y_b)), newshape=(-1, 1))
    x, y = sort_inputs(x, y, sort_axis=0)

    x_out = np.reshape(signal_generator.x, newshape=(-1, 1))

    return x_a, x_b, x, y_a, y_b, y, x_out, signal_generator.x, signal_generator.y
