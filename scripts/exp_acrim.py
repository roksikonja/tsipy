import datetime
import os

import gpflow as gpf
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf

import tsipy.fusion
from tsipy.fusion import (
    MultiWhiteKernel,
    build_sensor_labels,
    build_output_labels,
    concatenate_labels,
)
from utils import Constants as Const
from utils.data import transform_time_to_unit, create_results_dir
from utils.visualizer import pprint, plot_signals, plot_signals_and_confidence

if __name__ == "__main__":
    results_dir = create_results_dir("../results", "exp-acrim")
    t_field = "t"
    a_field, b_field, c_field = "a", "b", "c"

    # Load data
    data = pd.read_csv(
        os.path.join("../data", "ACRIM1_SATIRE_HF.txt"),
        delimiter=" ",
        header=None,
    )
    data = data.rename(
        columns={
            0: t_field,
            1: a_field,
            2: b_field,
            3: c_field,
        }
    )
    t_org = data[t_field].values.copy()
    data[t_field] = transform_time_to_unit(
        data[t_field] - data[t_field][0],
        x_label=Const.YEAR_UNIT,
        start=datetime.datetime(1980, 1, 1),
    )

    t = data[t_field].values
    t_a = t_b = t_c = t
    a = data[a_field].values
    b = data[b_field].values
    c = data[c_field].values

    pprint("data", data.shape)
    print(data.head().to_string() + "\n")

    fig, _ = plot_signals(
        [
            (t_a, a, r"$a$", False),
            (t_b, b, r"$b$", False),
            (t_c, c, r"$c$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        x_ticker=1,
        tight_layout=True,
    )
    fig.show()

    freqs_a, psd_a = scipy.signal.welch(a, fs=1.0, nperseg=1024)
    freqs_b, psd_b = scipy.signal.welch(b, fs=1.0, nperseg=1024)
    freqs_c, psd_c = scipy.signal.welch(c, fs=1.0, nperseg=1024)
    fig, _ = plot_signals(
        [
            (freqs_a, psd_a, r"$a$", False),
            (freqs_b, psd_b, r"$b$", False),
            (freqs_c, psd_c, r"$c$", False),
        ],
        results_dir=results_dir,
        title="signals_psd",
        legend="upper right",
        tight_layout=True,
        log_scale_x=True,
    )
    fig.show()

    """
        Data-fusion
    """
    gpf.config.set_default_float(np.float64)
    np.random.seed(Const.RANDOM_SEED)
    tf.random.set_seed(Const.RANDOM_SEED)

    pprint("t_a", t_a.shape, "a", a.shape)
    pprint("t_b", t_b.shape, "b", b.shape)
    pprint("t_c", t_c.shape, "c", c.shape)

    labels, t_labels = build_sensor_labels((t_a, t_b, t_c))
    s = np.hstack((a, b, c))
    t = np.hstack((t_a, t_b, t_c))
    t = concatenate_labels(t, t_labels)

    pprint("labels", labels)
    pprint("t_labels", t_labels.shape)
    pprint("t", t.shape)
    pprint("s", s.shape)

    t_out = t_a
    t_out_labels = build_output_labels(t_out)
    t_out = concatenate_labels(t_out, t_out_labels, sort_axis=0)

    pprint("t_out_labels", t_out_labels.shape)
    pprint("t_out", t_out.shape)

    """
        Kernel
    """
    # Signal kernel
    # matern_kernel = gpf.kernels.Matern52(active_dims=[0])  # Kernel for time dimension
    # matern_kernel = gpf.kernels.Matern32(active_dims=[0])  # Kernel for time dimension
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])  # Kernel for time dimension

    # Noise kernel
    # white_kernel = gpf.kernels.White(active_dims=[1])
    white_kernel = MultiWhiteKernel(
        labels=labels, active_dims=[1]
    )  # Kernel for sensor dimension

    # Kernel composite
    kernel = matern_kernel + white_kernel

    """
        Gaussian Process Model
    """
    fusion_model = tsipy.fusion.models.SVGPModel(kernel=kernel, num_inducing_pts=500)

    # Train
    fusion_model.fit(t, s, max_iter=20000, verbose=True)

    """
        Composite
    """
    # Predict
    s_out_mean, s_out_std = fusion_model(t_out)
    t_out = t_out[:, 0]

    pprint("t_out", t_out.shape)
    pprint("s_out_mean", s_out_mean.shape)
    pprint("s_out_std", s_out_std.shape)

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
        x_ticker=1,
    )
    fig.show()
    ax.scatter(
        t_a,
        a,
        label=r"$a$",
        s=Const.MARKER_SIZE,
    )
    ax.scatter(
        t_b,
        b,
        label=r"$b$",
        s=Const.MARKER_SIZE,
    )
    ax.scatter(
        t_c,
        c,
        label=r"$c$",
        s=Const.MARKER_SIZE,
    )
    fig.show()
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))

    freqs_s, psd_s = scipy.signal.welch(s_out_mean, fs=1.0, nperseg=1024)
    fig, ax = plot_signals(
        [
            (freqs_a, psd_a, r"$a$", False),
            (freqs_b, psd_b, r"$b$", False),
            (freqs_c, psd_c, r"$c$", False),
            (freqs_s, psd_s, r"$s$", False),
        ],
        results_dir=results_dir,
        title="signals_fused_psd",
        legend="upper right",
        tight_layout=True,
        log_scale_x=True,
    )
    ax.set_xscale("log")
    fig.show()

    """
        Training
    """
    elbo = fusion_model.iter_elbo
    fig, ax = plot_signals(
        [(np.arange(elbo.size), elbo, r"ELBO", False)],
        results_dir=results_dir,
        title="iter-elbo",
        legend="lower right",
        tight_layout=True,
    )
    fig.show()

    """
        Save
    """
    data_results = pd.DataFrame(
        {
            t_field + "_org": t_org,
            t_field: t_out,
            "s_out_mean": s_out_mean,
            "s_out_std": s_out_std,
        }
    )
    data_results.to_csv(os.path.join(results_dir, "data_results.csv"))
