import datetime
import os

import numpy as np
import pandas as pd


def load_data(data_dir, data_file):
    if data_file in os.listdir(data_dir):
        data = pd.read_hdf(os.path.join(data_dir, data_file), "table")
        return data

    return None


def mission_day_to_year(day):
    start = datetime.datetime(1996, 1, 1)
    years = start.year + day / 365.25

    return years


def is_integer(num):
    return isinstance(num, (int, np.int, np.int32, np.int64))


def downsample_signal(x, k=1):
    if not is_integer(k):
        raise Exception("Downsampling factor must be an integer.")
    if k > 1:
        return x[::k]
    else:
        return x


def time_output(t_a_nn, t_b_nn, n_out_per_unit=24):
    min_time = 0
    max_time = np.minimum(np.floor(t_a_nn.max()), np.floor(t_b_nn.max()))

    n_out = int(n_out_per_unit * (max_time - min_time) + 1)

    t_out = np.linspace(min_time, max_time, n_out).reshape(-1, 1)
    return t_out
