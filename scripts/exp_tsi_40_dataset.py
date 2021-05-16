"""
A script for fusing TSI signals given in a data file into a single composite using
Gaussian Processes.
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


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments specifying processing method."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", "-e", default="exp_tsi_40_datasets", type=str
    )

    parser.add_argument("--results_dir", default="../results", type=str)
    parser.add_argument("--data_dir", default="../data/tsi_40", type=str)

    parser.add_argument(
        "--dataset_file",
        default="ACRIM1_HF_full2.txt",
        type=str,
        help="Dataset file name.",
    )

    parser.add_argument("--random_seed", default=0, type=int)

    # Fusion Model
    parser.add_argument("--fusion_model", "-m", default="localgp", type=str)

    # Preprocess
    parser.add_argument("--normalization", "-n", action="store_false")
    parser.add_argument("--clipping", "-c", action="store_false")

    # SVGP
    parser.add_argument("--num_inducing_pts", "-n_ind_pts", default=100, type=int)
    parser.add_argument("--max_iter", default=1000, type=int)

    # Local GP
    parser.add_argument(
        "--pred_window",
        "-p_w",
        default=1.0,
        type=float,
        help="Width of prediction window in years.",
    )
    parser.add_argument(
        "--fit_window",
        "-f_w",
        default=3.0,
        type=float,
        help="Width of training window in years.",
    )

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


def load_dataset(dataset_path_: str) -> pd.DataFrame:
    data_ = pd.read_csv(
        dataset_path_,
        delimiter=" ",
        header=None,
    )

    columns_ = {
        j: chr(ord("a") + j - 1) if j != 0 else "t" for j in range(len(data_.columns))
    }
    data_ = data_.rename(columns=columns_)

    data_["t_org"] = data_["t"].values.copy()
    data_["t"] = transform_time_to_unit(
        data_["t"] - data_["t"][0],
        t_label="year",
        start=datetime.datetime(1980, 1, 1),
    )
    return data_


if __name__ == "__main__":
    args = parse_arguments()

    pprint_block("Experiment", args.experiment_name)

    dataset_path = os.path.join(args.data_dir, args.dataset_file)
    dataset = os.path.splitext(args.dataset_file)[0]
    data = load_dataset(dataset_path)

    results_dir = make_dir(
        os.path.join(args.results_dir, f"{args.experiment_name}_{dataset}")
    )
    results_dir = make_dir(os.path.join(results_dir, dataset))

    pprint_block("Dataset", dataset)

    t = data["t"].values
    signals = dict()
    psds = dict()
    for signal_name in data.columns:
        if signal_name not in ["t", "t_org"]:
            # signals[column] = data[column].values

            # Keep only not NaN values
            data_signal = data[["t", signal_name]].dropna()

            t_sig = data_signal["t"].values
            x_sig = data_signal[signal_name].values
            signals[signal_name] = (t_sig, x_sig)

            pprint("Signal", signal_name, level=0)
            pprint(f"- t_{signal_name}", t_sig.shape, level=1)
            pprint(f"- x_{signal_name}", x_sig.shape, level=1)

            n_per_seg = min(1024, x_sig.size)
            freqs, psd = scipy.signal.welch(x_sig, fs=1.0, nperseg=n_per_seg)
            psds[signal_name] = (freqs, psd)

    plot_signals(
        [
            (t_sig, x_sig, f"${signal_name}$", {})
            for signal_name, (t_sig, x_sig) in signals.items()
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        x_ticker=1,
        show=args.figure_show,
    )

    plot_signals(
        [
            (freqs, psd, f"${signal_name}$", {})
            for signal_name, (freqs, psd) in psds.items()
        ],
        results_dir=results_dir,
        title="signals_psd",
        legend="upper right",
        log_scale_x=True,
        show=args.figure_show,
    )

    pprint_block("Data Fusion")
    gpf.config.set_default_float(np.float64)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    signals_labeled = dict()
    for i, (signal_name, (t_sig, x_sig)) in enumerate(signals.items()):
        t_sig = build_and_concat_label_mask(t_sig, label=i + 1)
        signals_labeled[signal_name] = (t_sig, x_sig)

    t_out = build_and_concat_label_mask_output(t)

    # Concatenate signals and sort by x[:, 0]
    t = np.vstack(
        tuple(signals_labeled[signal_name][0] for signal_name in signals_labeled)
    )
    s = np.reshape(
        np.hstack(
            tuple(signals_labeled[signal_name][1] for signal_name in signals_labeled)
        ),
        newshape=(-1, 1),
    )
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
            pred_window_width=args.pred_window,
            fit_window_width=args.fit_window,
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
    fusion_model.fit(t, s, max_iter=args.max_iter, x_val=t_out, n_evals=5, verbose=True)

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
    for signal_name, (t_sig, x_sig) in signals.items():
        ax.scatter(
            t_sig,
            x_sig,
            label=f"${signal_name}$",
            s=3,
        )
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))
    if args.figure_show:
        fig.show()

    freqs_s, psd_s = scipy.signal.welch(s_out_mean, fs=1.0, nperseg=1024)
    fig, ax = plot_signals(
        [
            (freqs, psd, f"${signal_name}$", {})
            for signal_name, (freqs, psd) in psds.items()
        ]
        + [(freqs_s, psd_s, r"$s$", {})],
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
    data_results.to_csv(os.path.join(results_dir, "data_results.csv"), sep=" ")
