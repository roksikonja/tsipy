import numpy as np


def compute_exposure(
    x: np.ndarray,
    method: str = "num_measurements",
    x_mean: float = 1.0,
) -> np.ndarray:
    if method == "num_measurements":
        x = ~np.isnan(x)
        x = x.astype(np.float)
        x = np.cumsum(x)
    elif method == "exposure_sum":
        x = np.nan_to_num(x, nan=0.0, copy=True)  # NaNs to 0
        x = np.divide(x, x_mean, dtype=np.float)
        x = np.cumsum(x)
    else:
        raise ValueError("Invalid exposure method.")

    return x
