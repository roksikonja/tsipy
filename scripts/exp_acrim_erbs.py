import argparse
import datetime
import os

import gpflow as gpf
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf

import tsipy.fusion
from tsipy.fusion.utils import (
    build_labels,
    build_output_labels,
    concatenate_labels,
)
from tsipy.utils import pprint, pprint_block
from utils import Constants as Const
from utils.data import transform_time_to_unit, create_results_dir
from utils.visualizer import plot_signals, plot_signals_and_confidence


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="exp_acrim_erbs", type=str)

    # Fusion Model
    parser.add_argument("--fusion_model", default="svgp", type=str)

    # SVGP
    parser.add_argument("--num_inducing_pts", default=1000, type=int)
    parser.add_argument("--max_iter", default=8000, type=int)

    # Local GP
    parser.add_argument("--pred_window", default=2.0, type=float)
    parser.add_argument("--fit_window", default=6.0, type=float)

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--experiment_name", default="exp_acrim_erbs", type=str)
#
#     # Fusion Model
#     parser.add_argument("--fusion_model", default="localgp", type=str)
#
#     # SVGP
#     parser.add_argument("--num_inducing_pts", default=100, type=int)
#     parser.add_argument("--max_iter", default=2000, type=int)
#
#     # Local GP
#     parser.add_argument("--pred_window", default=1.0, type=float)
#     parser.add_argument("--fit_window", default=3.0, type=float)
#
#     # Visualize
#     parser.add_argument("-figure_show", action="store_false")
#     return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    results_dir = create_results_dir(
        "../results", f"{args.experiment_name}_{args.fusion_model}"
    )

    """
        Dataset
    """
    pprint_block("Dataset")
    t_field = "t"
    a_field, b_field, c_field = "a", "b", "c"

    # Load data
    data = pd.read_csv(
        os.path.join("../data", "VIRGOnew_ERBS_ACRIM2.txt"),
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
        start=datetime.datetime(1996, 1, 1),
    )

    t = data[t_field].values
    t_a = t_b = t_c = t
    a = data[a_field].values
    b = data[b_field].values
    c = data[c_field].values

    print(data, "\n")
    pprint("- data", data.shape)
    pprint("- " + t_field, t.shape)
    pprint("- " + a_field, a.shape, np.sum(~np.isnan(a)))
    pprint("- " + b_field, b.shape, np.sum(~np.isnan(b)))
    pprint("- " + c_field, c.shape, np.sum(~np.isnan(c)))

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

    """
        Data-fusion
    """
    pprint_block("Data Fusion")
    gpf.config.set_default_float(np.float64)
    np.random.seed(0)
    tf.random.set_seed(0)

    labels, t_labels = build_labels((t_a, t_b, t_c))
    s = np.hstack((a, b, c))
    t = np.hstack((t_a, t_b, t_c))
    t = concatenate_labels(t, t_labels)

    t_out = t_a
    t_out_labels = build_output_labels(t_out)
    # t has to be sorted!, sort_axis=0
    t_out = concatenate_labels(t_out, t_out_labels, sort_axis=0)

    pprint("labels", labels)
    pprint("t_labels", t_labels.shape)
    pprint("t", t.shape)
    pprint("s", s.shape)
    pprint("t_out_labels", t_out_labels.shape)
    pprint("t_out", t_out.shape)

    """
        Kernel
    """
    # Signal kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])  # Kernel for time dimension

    # Noise kernel
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(
        labels=labels, active_dims=[1]
    )  # Kernel for sensor dimension

    # Kernel composite
    kernel = matern_kernel + white_kernel

    """
        Gaussian Process Model
    """
    if args.fusion_model == "localgp":
        local_model = tsipy.fusion.SVGPModel(
            kernel=kernel,
            num_inducing_pts=args.num_inducing_pts,
            normalization=False,
            clipping=False,
        )

        local_windows = tsipy.fusion.local_gp.create_windows(
            t,
            s,
            pred_window=args.pred_window,
            fit_window=args.fit_window,
            normalization=False,
            clipping=False,
            verbose=True,
        )

        fusion_model = tsipy.fusion.LocalGPModel(
            model=local_model, windows=local_windows
        )

        # Train
        fusion_model.fit(max_iter=args.max_iter, verbose=True)

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

    """
        Composite
    """
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

    """
        Training
    """
    if args.fusion_model == "localgp":
        for i, window in enumerate(fusion_model.windows.list):
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
