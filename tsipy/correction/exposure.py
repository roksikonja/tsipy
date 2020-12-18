from enum import Enum, auto

import numpy as np


class ExposureMethod(Enum):
    NUM_MEASUREMENTS = auto()
    EXPOSURE_SUM = auto()


def compute_exposure(x, method, x_mean=1.0):
    if method == ExposureMethod.NUM_MEASUREMENTS:
        x = ~np.isnan(x)
        x = x.astype(np.float)
        x = np.cumsum(x)
    elif method == ExposureMethod.EXPOSURE_SUM:
        x = np.nan_to_num(x, nan=0.0, copy=True)  # NaNs to 0
        x = np.divide(x, x_mean, dtype=np.float)
        x = np.cumsum(x)
    else:
        raise ValueError("Invalid exposure method.")

    return x
