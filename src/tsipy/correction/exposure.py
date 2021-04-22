"""
Module implements a function for degradation correction.
"""

import numpy as np

__all__ = ["compute_exposure"]


def compute_exposure(
    x: np.ndarray,
    method: str = "num_measurements",
    x_mean: float = 1.0,
) -> np.ndarray:
    """Computes exposure of a given signal.

    In literature, exposure is found under an alias exposure time.
    If ``num_measurements`` method is selected, then the exposure equals to the number
    of measurements up to time ``t``.
    If ``exposure_sum`` method is selected, then the exposure equals to the cumulative
    sum of measurement values up to time ``t``. Works only for measurements with
    positive values. ``x_mean`` normalizes values before cumulative sum.

    Examples:
        >>> import numpy as np
        >>> x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        >>> compute_exposure(x, method="num_measurements")
        array([1., 1., 2., 3., 4.])
        >>> compute_exposure(x, method="exposure_sum")
        array([ 1.,  1.,  4.,  8., 13.])
    """
    if x.ndim != 1:
        raise ValueError(f"x of shape {x.shape} is not 1D array.")

    if method == "num_measurements":
        x = ~np.isnan(x)
        x = x.astype(np.float)
        return np.cumsum(x)
    elif method == "exposure_sum":
        x = np.nan_to_num(x, nan=0.0, copy=True)  # NaNs to 0
        x = np.divide(x, x_mean, dtype=np.float)
        return np.cumsum(x)

    raise ValueError("Invalid exposure method.")
