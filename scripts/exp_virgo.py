import argparse
import os

import gpflow as gpf
import numpy as np
import pandas as pd
import tensorflow as tf

import tsipy.correction
import tsipy.fusion
from tsipy.fusion.utils import (
    build_labels,
    build_output_labels,
    concatenate_labels,
)
from tsipy.utils import pprint, pprint_block, sort_inputs
from tsipy_utils.data import (
    make_dir,
    get_time_output,
    downsampling_indices_by_max_points,
)
from tsipy_utils.visualizer import (
    plot_signals,
    plot_signals_history,
    plot_signals_and_confidence,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", default="exp_virgo", type=str)

    # Degradation model
    parser.add_argument("--degradation_model", "-dm", default="explin", type=str)

    # SVGP
    parser.add_argument("--num_inducing_pts", "-n_ind_pts", default=1000, type=int)
    parser.add_argument("--max_iter", default=8000, type=int)

    # Visualize
    parser.add_argument("-figure_show", action="store_false")
    return parser.parse_args()


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "virgo":
        data = pd.read_hdf(os.path.join("../data", "virgo_2020.h5"), "table")
    else:
        raise ValueError("Dataset {} does not exist.".format(dataset_name))

    return data


if __name__ == "__main__":
    args = parse_arguments()

    pprint_block("Experiment", args.experiment_name)
    results_dir = make_dir(os.path.join("../results", args.experiment_name))

    pprint_block("Virgo Dataset", level=1)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Load data
    data = load_dataset("virgo")

    # Compute exposure
    e_a = tsipy.correction.compute_exposure(data["a"].values)
    e_b = tsipy.correction.compute_exposure(data["b"].values)
    max_e = max(np.max(e_a), np.max(e_b))
    e_a /= max_e
    e_b /= max_e
    data["e_a"] = e_a
    data["e_b"] = e_b

    # Channel measurements
    data_a = data[["t", "a", "e_a"]].dropna()
    data_b = data[["t", "b", "e_b"]].dropna()

    t_a_nn, t_b_nn = data_a["t"].values, data_b["t"].values
    a_nn, b_nn = data_a["a"].values, data_b["b"].values
    e_a_nn, e_b_nn = data_a["e_a"].values, data_b["e_b"].values

    pprint("Signal", level=0)
    pprint("- t_a_nn", t_a_nn.shape, level=1)
    pprint("- a_nn", a_nn.shape, level=1)
    pprint("- e_a_nn", e_a_nn.shape, level=1)

    pprint("Signal", level=0)
    pprint("- t_b_nn", t_b_nn.shape, level=1)
    pprint("- b_nn", b_nn.shape, level=1)
    pprint("- e_b_nn", e_b_nn.shape, level=1)

    # Mutual measurements
    pprint_block("Simultaneous measurements", level=1)
    data_m = data[["t", "a", "b", "e_a", "e_b"]].dropna()
    t_m = data_m["t"].values
    a_m, b_m = data_m["a"].values, data_m["b"].values
    e_a_m, e_b_m = data_m["e_a"].values, data_m["e_b"].values

    pprint("Signal", level=0)
    pprint("- t_m", t_m.shape, level=1)
    pprint("- a_m", a_m.shape, level=1)
    pprint("- e_a_m", e_a_m.shape, level=1)

    pprint("Signal", level=0)
    pprint("- t_m", t_m.shape, level=1)
    pprint("- b_m", b_m.shape, level=1)
    pprint("- e_b_m", e_b_m.shape, level=1)

    fig, _ = plot_signals(
        [
            (t_a_nn, a_nn, r"$a$", False),
            (t_b_nn, b_nn, r"$b$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        x_ticker=4,
        y_lim=[1357, 1369],
    )
    if args.figure_show:
        fig.show()

    pprint_block("Degradation Correction", level=1)
    degradation_model = tsipy.correction.load_model(args.degradation_model)
    degradation_model.initial_fit(a_m, b_m, e_a_m)

    a_m_c, b_m_c, degradation_model, history = tsipy.correction.correct_degradation(
        t_m,
        a_m,
        e_a_m,
        b_m,
        e_b_m,
        model=degradation_model,
        verbose=True,
    )

    d_a_c = degradation_model(e_a_nn)
    d_b_c = degradation_model(e_b_nn)
    a_c_nn = np.divide(a_nn, d_a_c)
    b_c_nn = np.divide(b_nn, d_b_c)

    fig, _ = plot_signals(
        [
            (t_m, a_m_c, r"$a_c$", False),
            (t_m, b_m_c, r"$b_c$", False),
        ],
        results_dir=results_dir,
        title="signals_corrected",
        legend="upper right",
        x_ticker=4,
    )
    if args.figure_show:
        fig.show()

    fig, _ = plot_signals(
        [
            (t_a_nn, d_a_c, r"$d(e_a(t))$", False),
            (t_b_nn, d_b_c, r"$d(e_b(t))$", False),
        ],
        results_dir=results_dir,
        title="degradation",
        legend="lower left",
        x_ticker=4,
    )
    if args.figure_show:
        fig.show()

    fig, _ = plot_signals_history(
        t_m,
        [
            [
                (signals[0], r"$a_{}$".format(i)),
                (signals[1], r"$b_{}$".format(i)),
            ]
            for i, signals in enumerate(history[:4])
        ],
        results_dir,
        title="correction-history",
        n_rows=2,
        n_cols=2,
        x_ticker=4,
    )
    if args.figure_show:
        fig.show()

    pprint_block("Data Fusion", level=1)
    gpf.config.set_default_float(np.float64)

    labels, t_labels = build_labels([t_a_nn, t_b_nn])
    s = np.reshape(np.hstack((a_c_nn, b_c_nn)), newshape=(-1, 1))
    t = np.hstack((t_a_nn, t_b_nn))
    t = concatenate_labels(t, t_labels)
    t, s = sort_inputs(t, s, sort_axis=0)

    t_out = get_time_output([t_a_nn, t_b_nn], n_out_per_unit=365 * 24)
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

    # Kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(labels=labels, active_dims=[1])
    kernel = matern_kernel + white_kernel

    fusion_model = tsipy.fusion.models_gp.SVGPModel(
        kernel=kernel,
        num_inducing_pts=args.num_inducing_pts,
    )

    # Train
    fusion_model.fit(t, s, max_iter=args.max_iter, verbose=True)

    # Predict
    s_out_mean, s_out_std = fusion_model(t_out)
    t_out = t_out[:, 0]

    pprint("Output Signal", level=0)
    pprint("- t_out", t_out.shape, level=1)
    pprint("- s_out_mean", s_out_mean.shape, level=1)
    pprint("- s_out_std", s_out_std.shape, level=1)

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
        x_ticker=4,
        y_lim=[1362, 1369],
    )
    if args.figure_show:
        fig.show()
    indices_a = downsampling_indices_by_max_points(
        t_a_nn, max_points=20_000
    )  # Downsample signal a for plotting
    ax.scatter(
        t_a_nn[indices_a],
        a_c_nn[indices_a],
        label=r"$a$",
        s=3,
    )
    ax.scatter(
        t_b_nn,
        b_c_nn,
        label=r"$b$",
        s=3,
    )
    if args.figure_show:
        fig.show()
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))

    elbo = fusion_model.iter_elbo
    fig, ax = plot_signals(
        [(np.arange(elbo.size), elbo, r"ELBO", False)],
        results_dir=results_dir,
        title="iter_elbo",
        legend="lower right",
    )
    if args.figure_show:
        fig.show()
