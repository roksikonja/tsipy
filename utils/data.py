import datetime
import os

import numpy as np
import pandas as pd

from .constants import Constants as Const


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def create_results_dir(results_dir_path, dir_name="results"):
    results_dir = make_dir(
        os.path.join(
            results_dir_path,
            datetime.datetime.now().strftime(f"%m-%d_%H-%M-%S_{dir_name}"),
        )
    )
    return results_dir


def load_data(data_dir, data_file):
    if data_file in os.listdir(data_dir):
        data = pd.read_hdf(os.path.join(data_dir, data_file), "table")
        return data

    return None


def is_integer(num):
    return isinstance(num, (int, np.int, np.int32, np.int64))


def downsample_signal(x, k=1):
    if not is_integer(k):
        raise Exception("Downsampling factor must be an integer.")
    if k > 1:
        return x[::k]
    else:
        return x


def downsampling_indices_by_max_points(x, max_points=1e5):
    indices = np.ones_like(x, dtype=np.bool)
    if x.shape[0] > max_points:
        downsampling_factor = x.shape[0] // int(max_points)

        indices = np.zeros_like(x, dtype=np.bool)
        indices[::downsampling_factor] = True

    return indices


def mission_day_to_year(day, start):
    years = start.year + day / 365.25

    return years


def transform_time_to_unit(
    t, x_label=Const.YEAR_UNIT, start=datetime.datetime(1996, 1, 1)
):
    if x_label == Const.YEAR_UNIT:
        t = np.array([mission_day_to_year(t_, start) for t_ in t])

    return t


def get_time_output(t_nns, n_out_per_unit=24, min_time=None, max_time=None):
    if not min_time:
        min_time = np.max([np.min(t_nn) for t_nn in t_nns])

    if not max_time:
        max_time = np.min([np.max(t_nn) for t_nn in t_nns])

    n_out = int(n_out_per_unit * (max_time - min_time) + 1)

    t_out = np.linspace(min_time, max_time, n_out)
    return t_out
