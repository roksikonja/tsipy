"""
A script for processing VIRGO level-1 TSI dataset.

First, the script corrects signals from instruments PMODV6-A and
PMODV6-B for degradation. Then, it produces a TSI composite using
Gaussian Processes.
"""
import argparse
import os

import gpflow as gpf
import numpy as np
import pandas as pd
import tensorflow as tf

from tsipy.correction import compute_exposure, correct_degradation, load_model
from tsipy.fusion import SVGPModel
from tsipy.fusion.kernels import MultiWhiteKernel
from tsipy.fusion.utils import (
    build_and_concat_label_mask,
    build_and_concat_label_mask_output,
)
from tsipy.utils import (
    downsampling_indices_by_max_points,
    get_time_output,
    make_dir,
    plot_signals,
    plot_signals_and_confidence,
    plot_signals_history,
    pprint,
    pprint_block,
    sort_inputs,
)


def parse_arguments():
    """Parses command line arguments specifying processing method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", default="exp_virgo", type=str)

    # Degradation model
    parser.add_argument("--degradation_model", "-dm", default="explin", type=str)

    # SVGP
    parser.add_argument("--num_inducing_pts", "-n_ind_pts", default=1000, type=int)
    parser.add_argument("--max_iter", default=8000, type=int)

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


def load_dataset(dataset_name: str):
    """Loads the VIRGO dataset."""
    if dataset_name == "virgo":
        data_frame = pd.read_hdf(
            os.path.join("../data", "virgo_level1_2020.h5"), "table"
        )
        data_frame = data_frame.rename(
            columns={"TIME": "t", "PMO6V-A": "a", "PMO6V-B": "b"}
        )
        data_frame.drop(columns=["TEMPERATURE"], inplace=True)
        return data_frame
    elif dataset_name == "virgo_2020":
        return pd.read_hdf(os.path.join("../data", "virgo_2020.h5"), "table")
    else:
        raise ValueError("Dataset {} does not exist.".format(dataset_name))


if __name__ == "__main__":
    args = parse_arguments()

    pprint_block("Experiment", args.experiment_name)
    results_dir = make_dir(os.path.join("../results", args.experiment_name))

    pprint_block("Virgo Dataset", level=1)
    np.random.seed(0)
    tf.random.set_seed(0)

    # Load data
    data = load_dataset("virgo")
    print(data.head())

    # Compute exposure
    e_a = compute_exposure(data["a"].values)
    e_b = compute_exposure(data["b"].values)
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

    plot_signals(
        [
            (t_a_nn, a_nn, r"$a$", {}),
            (t_b_nn, b_nn, r"$b$", {}),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        y_lim=[1357, 1369],
        show=args.figure_show,
    )

    pprint_block("Degradation Correction", level=1)
    degradation_model = load_model(args.degradation_model)
    degradation_model.initial_fit(x_a=e_a_m, y_a=a_m, y_b=b_m)

    a_m_c, b_m_c, degradation_model, history = correct_degradation(
        t_m=t_m,
        a_m=a_m,
        e_a_m=e_a_m,
        b_m=b_m,
        e_b_m=e_b_m,
        model=degradation_model,
        verbose=True,
    )

    d_a_c = degradation_model(e_a_nn)
    d_b_c = degradation_model(e_b_nn)
    a_c_nn = np.divide(a_nn, d_a_c)
    b_c_nn = np.divide(b_nn, d_b_c)

    pprint("Corrected Signal", level=0)
    pprint("- a_c_nn", a_c_nn.shape, level=1)
    pprint("- d_a_c", d_a_c.shape, level=1)

    pprint("Corrected Signal", level=0)
    pprint("- b_c_nn", b_c_nn.shape, level=1)
    pprint("- d_b_c", d_b_c.shape, level=1)

    plot_signals(
        [
            (t_m, a_m_c, "$a_c$", {}),
            (t_m, b_m_c, "$b_c$", {}),
        ],
        results_dir=results_dir,
        title="signals_corrected",
        legend="upper right",
        show=args.figure_show,
    )

    plot_signals(
        [
            (t_a_nn, d_a_c, "$d(e_a(t))$", {}),
            (t_b_nn, d_b_c, "$d(e_b(t))$", {}),
        ],
        results_dir=results_dir,
        title="degradation",
        legend="lower left",
        show=args.figure_show,
    )

    plot_signals_history(
        t_m,
        [
            [
                (signals.a, "$a_{}$".format(i)),
                (signals.b, "$b_{}$".format(i)),
            ]
            for i, signals in enumerate(history[:4])
        ],
        results_dir,
        title="correction-history",
        n_rows=2,
        n_cols=2,
        tight_layout=True,
        show=args.figure_show,
    )

    pprint_block("Data Fusion", level=1)
    gpf.config.set_default_float(np.float64)

    t_a_nn = build_and_concat_label_mask(t_a_nn, label=1)
    t_b_nn = build_and_concat_label_mask(t_b_nn, label=2)
    t_out = get_time_output([t_a_nn, t_b_nn], n_per_unit=24)
    t_out = build_and_concat_label_mask_output(t_out)

    # Concatenate signals and sort by x[:, 0]
    t = np.vstack((t_a_nn, t_b_nn))
    s = np.reshape(np.hstack((a_c_nn, b_c_nn)), newshape=(-1, 1))
    t, s = sort_inputs(t, s, sort_axis=0)

    pprint("Signals", level=0)
    pprint("- t", t.shape, level=1)
    pprint("- s", s.shape, level=1)

    # Kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = MultiWhiteKernel(labels=(1, 2), active_dims=[1])
    kernel = matern_kernel + white_kernel

    fusion_model = SVGPModel(
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
        y_lim=[1362, 1369],
    )
    indices_a = downsampling_indices_by_max_points(
        t_a_nn[:, 0], max_points=20_000
    )  # Downsample signal a for plotting
    ax.scatter(
        t_a_nn[indices_a, 0],
        a_c_nn[indices_a],
        label=r"$a$",
        s=3,
    )
    ax.scatter(
        t_b_nn[:, 0],
        b_c_nn,
        label=r"$b$",
        s=3,
    )
    if args.figure_show:
        fig.show()
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))

    elbo = fusion_model.iter_elbo
    fig, ax = plot_signals(
        [(np.arange(elbo.size), elbo, r"ELBO", {})],  # type: ignore
        results_dir=results_dir,
        title="iter_elbo",
        legend="lower right",
        show=args.figure_show,
    )
