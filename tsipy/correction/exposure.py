from enum import Enum, auto

import numpy as np


class ExposureMethod(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


def compute_exposure(x, method):
    # NaNs to 0
    x = np.nan_to_num(x, nan=0.0, copy=True)

    if method == ExposureMethod.NUM_MEASUREMENTS:
        x = np.greater(x, 0, dtype=np.float)
        x = np.cumsum(x)

    elif method == ExposureMethod.EXPOSURE_SUM:
        x = np.divide(x, dtype=np.float)
        x = np.cumsum(x)
    else:
        raise ValueError("Invalid exposure method.")

    return x
