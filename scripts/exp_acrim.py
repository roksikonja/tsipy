import argparse
import datetime
import os

import gpflow as gpf
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf

import tsipy.fusion
import tsipy.fusion
from tsipy.fusion.utils import (
    build_labels,
    build_output_labels,
    concatenate_labels,
)
from tsipy.utils import pprint, pprint_block, sort_inputs
from tsipy_utils.data import transform_time_to_unit, make_dir
from tsipy_utils.visualizer import plot_signals_and_confidence, plot_signals


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", default="demo_fusion", type=str)

    # Fusion Model
    parser.add_argument("--fusion_model", "-m", default="svgp", type=str)

    # Preprocess
    parser.add_argument("--normalization", "-n", action="store_false")
    parser.add_argument("--clipping", "-c", action="store_false")

    # SVGP
    parser.add_argument("--num_inducing_pts", "-n_ind_pts", default=1000, type=int)
    parser.add_argument("--max_iter", default=8000, type=int)

    # Local GP
    parser.add_argument("--pred_window", "-p_w", default=0.2, type=float)
    parser.add_argument("--fit_window", "-f_w", default=0.6, type=float)

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    pprint_block("Experiment", args.experiment_name)
    results_dir = make_dir(os.path.join("../results", args.experiment_name))

    pprint_block("Dataset")
    # Load data
    data = pd.read_csv(
        os.path.join("../data", "ACRIM1_SATIRE_HF.txt"),
        delimiter=" ",
        header=None,
    )
    data = data.rename(
        columns={
            0: "t",
            1: "a",
            2: "b",
            3: "c",
        }
    )
    t_org = data["t"].values.copy()
    data["t"] = transform_time_to_unit(
        data["t"] - data["t"][0],
        x_label="year",
        start=datetime.datetime(1980, 1, 1),
    )

    t_a = t_b = t_c = data["t"].values
    a = data["a"].values
    b = data["b"].values
    c = data["c"].values

    pprint("Signal", level=0)
    pprint("- t_a", t_a.shape, level=1)
    pprint("- a", a.shape, level=1)

    pprint("Signal", level=0)
    pprint("- t_b", t_b.shape, level=1)
    pprint("- b", b.shape, level=1)

    pprint("Signal", level=0)
    pprint("- t_c", t_c.shape, level=1)
    pprint("- c", c.shape, level=1)

    plot_signals(
        [
            (t_a, a, r"$a$", False),
            (t_b, b, r"$b$", False),
            (t_c, c, r"$c$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        x_ticker=1,
        show=args.figure_show,
    )

    freqs_a, psd_a = scipy.signal.welch(a, fs=1.0, nperseg=1024)
    freqs_b, psd_b = scipy.signal.welch(b, fs=1.0, nperseg=1024)
    freqs_c, psd_c = scipy.signal.welch(c, fs=1.0, nperseg=1024)
    plot_signals(
        [
            (freqs_a, psd_a, r"$a$", False),
            (freqs_b, psd_b, r"$b$", False),
            (freqs_c, psd_c, r"$c$", False),
        ],
        results_dir=results_dir,
        title="signals_psd",
        legend="upper right",
        log_scale_x=True,
        show=args.figure_show,
    )

    pprint_block("Data Fusion")
    gpf.config.set_default_float(np.float64)
    np.random.seed(0)
    tf.random.set_seed(0)

    labels, t_labels = build_labels([t_a, t_b, t_c])
    s = np.reshape(np.hstack((a, b, c)), newshape=(-1, 1))
    t = np.hstack((t_a, t_b, t_c))
    t = concatenate_labels(t, t_labels)
    t, s = sort_inputs(t, s, sort_axis=0)

    t_out = t
    t_out_labels = build_output_labels(t_out)
    t_out = concatenate_labels(t_out, t_out_labels)

    pprint("Signals", level=0)
    pprint("- t", t.shape, level=1)
    pprint("- s", s.shape, level=1)

    pprint("Signal", level=0)
    pprint("- labels", labels, level=1)
    pprint("- t_labels", t_labels.shape, level=1)
    pprint("- t_out_labels", t_out_labels.shape, level=1)
    pprint("- t_out", t_out.shape, level=1)

    # Signal kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])  # Kernel for time dimension

    # Noise kernel
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(
        labels=labels, active_dims=[1]
    )  # Kernel for sensor dimension

    # Kernel composite
    kernel = matern_kernel + white_kernel

    if args.fusion_model == "localgp":
        local_model = tsipy.fusion.SVGPModel(
            kernel=kernel,
            num_inducing_pts=args.num_inducing_pts,
        )

        local_windows = tsipy.fusion.local_gp.create_windows(
            t,
            s,
            pred_window=args.pred_window,
            fit_window=args.fit_window,
            verbose=True,
        )

        fusion_model = tsipy.fusion.LocalGPModel(
            model=local_model,
            normalization=args.normalization,
            clipping=args.clipping,
        )

        # Train
        fusion_model.fit(windows=local_windows, max_iter=args.max_iter, verbose=True)

        # Predict
        pprint_block("Inference", level=2)
        s_out_mean, s_out_std = fusion_model(t_out, verbose=True)
    else:
        fusion_model = tsipy.fusion.SVGPModel(
            kernel=kernel, num_inducing_pts=args.num_inducing_pts
        )

        # Train
        fusion_model.fit(t, s, max_iter=args.max_iter)

        # Predict
        pprint_block("Inference", level=2)
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
    ax.scatter(
        t_a,
        a,
        label=r"$a$",
        s=3,
    )
    ax.scatter(
        t_b,
        b,
        label=r"$b$",
        s=3,
    )
    ax.scatter(
        t_c,
        c,
        label=r"$c$",
        s=3,
    )
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))
    if args.figure_show:
        fig.show()

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
        log_scale_x=True,
    )
    ax.set_xscale("log")
    fig.savefig(os.path.join(results_dir, "signals_fused_psd_log"))
    if args.figure_show:
        fig.show()

    if args.fusion_model == "localgp":
        for i, window in enumerate(fusion_model.windows):
            elbo = window.model.iter_elbo
            plot_signals(
                [(np.arange(elbo.size), elbo, r"ELBO", False)],
                results_dir=results_dir,
                title=f"iter_elbo_w{i}",
                legend="lower right",
                show=args.figure_show,
            )
    else:
        elbo = fusion_model.iter_elbo
        plot_signals(
            [(np.arange(elbo.size), elbo, r"ELBO", False)],
            results_dir=results_dir,
            title="iter_elbo",
            legend="lower right",
            show=args.figure_show,
        )

    data_results = pd.DataFrame(
        {
            "t" + "_org": t_org,
            "t": t_out,
            "s_out_mean": s_out_mean,
            "s_out_std": s_out_std,
        }
    )
    data_results.to_csv(os.path.join(results_dir, "data_results.csv"))
