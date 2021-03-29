import datetime
import os
from typing import Any, List, Union

import numpy as np

__all__ = [
    "make_dir",
    "create_results_dir",
    "is_integer",
    "downsample_signal",
    "downsampling_indices_by_max_points",
    "mission_day_to_year",
    "transform_time_to_unit",
    "get_time_output",
]


def make_dir(directory: str) -> str:
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def create_results_dir(results_dir_path: str, dir_name: str = "results") -> str:
    results_dir = make_dir(
        os.path.join(
            results_dir_path,
            datetime.datetime.now().strftime(f"%m-%d_%H-%M-%S_{dir_name}"),
        )
    )
    return results_dir


def is_integer(num: Any) -> bool:
    return isinstance(num, (int, np.int, np.int32, np.int64))


def downsample_signal(x: np.ndarray, k: int = 1) -> np.ndarray:
    if not is_integer(k):
        raise Exception("Downsampling factor must be an integer.")
    if k > 1:
        return x[::k]
    else:
        return x


def downsampling_indices_by_max_points(
    x: np.ndarray, max_points: int = 100_000
) -> np.ndarray:
    indices = np.ones_like(x, dtype=np.bool)
    if x.shape[0] > max_points:
        downsampling_factor = x.shape[0] // int(max_points)

        indices = np.zeros_like(x, dtype=np.bool)
        indices[::downsampling_factor] = True

    return indices


def mission_day_to_year(day: float, start: datetime.datetime) -> float:
    years = start.year + day / 365.25

    return years


def transform_time_to_unit(
    t: np.ndarray,
    x_label: str = "year",
    start: datetime.datetime = datetime.datetime(1996, 1, 1),
) -> np.ndarray:
    if x_label == "year":
        t = np.array([mission_day_to_year(t_, start) for t_ in t])

    return t


def get_time_output(
    t_nns: List[np.ndarray],
    n_out_per_unit: int = 24,
    min_time: Union[np.ndarray, float] = None,
    max_time: Union[np.ndarray, float] = None,
) -> np.ndarray:
    if min_time is None:
        min_time = np.max([np.min(t_nn) for t_nn in t_nns])

    if max_time is None:
        max_time = np.min([np.max(t_nn) for t_nn in t_nns])

    n_out = int(n_out_per_unit * (max_time - min_time) + 1)

    t_out = np.linspace(min_time, max_time, n_out)
    return t_out
