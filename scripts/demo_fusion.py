import argparse
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf

import tsipy.fusion
from tsipy.correction.generator import SignalGenerator
from tsipy.fusion.utils import (
    build_labels,
    build_output_labels,
    concatenate_labels,
)
from tsipy.utils import pprint, pprint_block, sort_inputs
from tsipy_utils.data import make_dir
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
    np.random.seed(0)
    tf.random.set_seed(0)

    # Generate Brownian motion signal
    signal_generator = SignalGenerator(add_degradation=False)

    t_a, a = signal_generator["a"]
    t_b, b = signal_generator["b"]

    pprint("Signal", level=0)
    pprint("- t_a", t_a.shape, level=1)
    pprint("- a", a.shape, level=1)

    pprint("Signal", level=0)
    pprint("- t_b", t_b.shape, level=1)
    pprint("- b", b.shape, level=1)

    plot_signals(
        [
            (t_a, a, "$a$", False),
            (t_b, b, "$b$", False),
            (signal_generator.x, signal_generator.y, "$s$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        show=args.figure_show,
    )

    pprint_block("Data Fusion")
    gpf.config.set_default_float(np.float64)
    np.random.seed(0)
    tf.random.set_seed(0)

    labels, t_labels = build_labels([t_a, t_b])
    s = np.reshape(np.hstack((a, b)), newshape=(-1, 1))
    t = np.hstack((t_a, t_b))
    t = concatenate_labels(t, t_labels)
    t, s = sort_inputs(t, s, sort_axis=0)

    t_out = signal_generator.x
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
        )

        fusion_model = tsipy.fusion.local_gp.LocalGPModel(
            model=local_model,
            normalization=args.normalization,
            clipping=args.clipping,
        )

        # Train
        pprint_block("Training", level=2)
        fusion_model.fit(windows=local_windows, max_iter=args.max_iter)

        # Predict
        pprint_block("Inference", level=2)
        s_out_mean, s_out_std = fusion_model(t_out, verbose=True)
    else:
        fusion_model = tsipy.fusion.SVGPModel(
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

    pprint_block("Results")
    t_out = t_out[:, 0]

    pprint("- t_out", t_out.shape, level=1)
    pprint("- s_out_mean", s_out_mean.shape, level=1)
    pprint("- s_out_std", s_out_std.shape, level=1)

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
    )
    ax.scatter(
        t_a,
        a,
        label="$a$",
        s=3,
    )
    ax.scatter(
        t_b,
        b,
        label="$b$",
        s=3,
    )
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))
    if args.figure_show:
        fig.show()

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused_s",
    )
    ax.plot(signal_generator.x, signal_generator.y, label="$s$")
    fig.savefig(os.path.join(results_dir, "signals_fused_s"))
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