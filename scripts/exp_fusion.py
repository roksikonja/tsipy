"""
A script for fusing three TSI signals into a single composite using Gaussian Processes.
"""
import argparse
import datetime
import os

import gpflow as gpf
import numpy as np
import pandas as pd
import scipy.signal
import tensorflow as tf

from tsipy.fusion import LocalGPModel, SVGPModel
from tsipy.fusion.kernels import MultiWhiteKernel
from tsipy.fusion.utils import (
    build_and_concat_label_mask,
    build_and_concat_label_mask_output,
)
from tsipy.utils import (
    make_dir,
    plot_signals,
    plot_signals_and_confidence,
    pprint,
    pprint_block,
    sort_inputs,
    transform_time_to_unit,
)


def parse_arguments():
    """Parses command line arguments specifying processing method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", default="exp_fusion", type=str)
    parser.add_argument(
        "--dataset_name",
        "-d",
        default="acrim",
        type=str,
        help="Choices: acrim_erbs, acrim.",
    )

    # Fusion Model
    parser.add_argument("--fusion_model", "-m", default="svgp", type=str)

    # Preprocess
    parser.add_argument("--normalization", "-n", action="store_false")
    parser.add_argument("--clipping", "-c", action="store_false")

    # SVGP
    parser.add_argument("--num_inducing_pts", "-n_ind_pts", default=100, type=int)
    parser.add_argument("--max_iter", default=1000, type=int)

    # Local GP
    parser.add_argument("--pred_window", "-p_w", default=0.2, type=float)
    parser.add_argument("--fit_window", "-f_w", default=0.6, type=float)

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Loads the ACRIM or ACRIM+ERBS dataset."""
    if dataset_name == "acrim_erbs":
        data_ = pd.read_csv(
            os.path.join("../data", "VIRGOnew_ERBS_ACRIM2.txt"),
            delimiter=" ",
            header=None,
        )
        data_ = data_.rename(
            columns={
                0: "t",
                1: "a",
                2: "b",
                3: "c",
            }
        )

        data_["t_org"] = data_["t"].values.copy()
        data_["t"] = transform_time_to_unit(
            data_["t"] - data_["t"][0],
            t_label="year",
            start=datetime.datetime(1996, 1, 1),
        )
    elif dataset_name == "acrim":
        data_ = pd.read_csv(
            os.path.join("../data", "ACRIM1_SATIRE_HF.txt"),
            delimiter=" ",
            header=None,
        )
        data_ = data_.rename(
            columns={
                0: "t",
                1: "a",
                2: "b",
                3: "c",
            }
        )
        data_["t_org"] = data_["t"].values.copy()
        data_["t"] = transform_time_to_unit(
            data_["t"] - data_["t"][0],
            t_label="year",
            start=datetime.datetime(1980, 1, 1),
        )
    else:
        raise ValueError("Dataset {} does not exist.".format(dataset_name))

    return data_


if __name__ == "__main__":
    args = parse_arguments()

    pprint_block("Experiment", args.experiment_name)
    results_dir = make_dir(
        os.path.join("../results", f"{args.experiment_name}_{args.dataset_name}")
    )

    pprint_block("Dataset")
    data = load_dataset(args.dataset_name)
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
            (t_a, a, r"$a$", {}),
            (t_b, b, r"$b$", {}),
            (t_c, c, r"$c$", {}),
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
            (freqs_a, psd_a, r"$a$", {}),
            (freqs_b, psd_b, r"$b$", {}),
            (freqs_c, psd_c, r"$c$", {}),
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

    t_a = build_and_concat_label_mask(t_a, label=1)
    t_b = build_and_concat_label_mask(t_b, label=2)
    t_c = build_and_concat_label_mask(t_c, label=3)
    t_out = build_and_concat_label_mask_output(t_a)

    # Concatenate signals and sort by x[:, 0]
    t = np.vstack((t_a, t_b, t_c))
    s = np.reshape(np.hstack((a, b, c)), newshape=(-1, 1))
    t, s = sort_inputs(t, s, sort_axis=0)

    pprint("Signals", level=0)
    pprint("- t", t.shape, level=1)
    pprint("- s", s.shape, level=1)

    # Kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = MultiWhiteKernel(labels=(1, 2, 3), active_dims=[1])
    kernel = matern_kernel + white_kernel

    if args.fusion_model == "localgp":
        local_model = SVGPModel(
            kernel=kernel,
            num_inducing_pts=args.num_inducing_pts,
        )
        fusion_model = LocalGPModel(
            model=local_model,
            pred_window_width=1.0,
            fit_window_width=1.0,
            normalization=args.normalization,
            clipping=args.clipping,
        )
    else:
        fusion_model = SVGPModel(  # type: ignore
            kernel=kernel,
            num_inducing_pts=args.num_inducing_pts,
            normalization=args.normalization,
            clipping=args.clipping,
        )

    # Train
    pprint_block("Training", level=2)
    fusion_model.fit(t, s, max_iter=args.max_iter, x_val=t_out, n_evals=5)

    # Predict
    pprint_block("Inference", level=2)
    s_out_mean, s_out_std = fusion_model(t_out)

    pprint("Output Signal", level=0)
    pprint("- t_out", t_out.shape, level=1)
    pprint("- s_out_mean", s_out_mean.shape, level=1)
    pprint("- s_out_std", s_out_std.shape, level=1)

    fig, ax = plot_signals_and_confidence(
        [(t_out[:, 0], s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
        x_ticker=1,
    )
    ax.scatter(
        t_a[:, 0],
        a,
        label=r"$a$",
        s=3,
    )
    ax.scatter(
        t_b[:, 0],
        b,
        label=r"$b$",
        s=3,
    )
    ax.scatter(
        t_c[:, 0],
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
            (freqs_a, psd_a, r"$a$", {}),
            (freqs_b, psd_b, r"$b$", {}),
            (freqs_c, psd_c, r"$c$", {}),
            (freqs_s, psd_s, r"$s$", {}),
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
                [(np.arange(elbo.size), elbo, r"ELBO", {})],
                results_dir=results_dir,
                title=f"iter_elbo_w{i}",
                legend="lower right",
                show=args.figure_show,
            )
    else:
        elbo = fusion_model.iter_elbo  # type: ignore
        plot_signals(
            [(np.arange(elbo.size), elbo, r"ELBO", {})],
            results_dir=results_dir,
            title="iter_elbo",
            legend="lower right",
            show=args.figure_show,
        )

    data_results = pd.DataFrame(
        {
            "t_org": data["t_org"].values,
            "t": t_out[:, 0],
            "s_out_mean": s_out_mean,
            "s_out_std": s_out_std,
        }
    )
    data_results.to_csv(os.path.join(results_dir, "data_results.csv"))
