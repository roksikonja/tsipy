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


def mission_day_to_year(day):
    start = datetime.datetime(1996, 1, 1)
    years = start.year + day / 365.25

    return years


def transform_time_to_unit(t, x_label=Const.YEAR_UNIT):
    if x_label == Const.YEAR_UNIT:
        t = np.array(list(map(mission_day_to_year, t)))

    return t


def get_time_output(t_a_nn, t_b_nn, n_out_per_unit=24):
    min_time = 0
    max_time = np.minimum(np.floor(t_a_nn.max()), np.floor(t_b_nn.max()))

    n_out = int(n_out_per_unit * (max_time - min_time) + 1)

    t_out = np.linspace(min_time, max_time, n_out)
    return t_out
