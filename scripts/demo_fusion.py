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
from tsipy_utils.data import create_results_dir
from tsipy_utils.visualizer import plot_signals_and_confidence, plot_signals


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="demo_fusion", type=str)

    # Fusion Model
    # parser.add_argument("--fusion_model", default="svgp", type=str)
    parser.add_argument("--fusion_model", default="localgp", type=str)

    # Preprocess
    # parser.add_argument("--normalization", action="store_false")
    # parser.add_argument("--clipping", action="store_false")
    parser.add_argument("--normalization", action="store_true")
    parser.add_argument("--clipping", action="store_true")

    # SVGP
    # parser.add_argument("--num_inducing_pts", default=1000, type=int)
    # parser.add_argument("--max_iter", default=8000, type=int)
    parser.add_argument("--num_inducing_pts", default=100, type=int)
    parser.add_argument("--max_iter", default=200, type=int)

    # Local GP
    parser.add_argument("--pred_window", default=1.0, type=float)
    parser.add_argument("--fit_window", default=1.0, type=float)

    # Visualize
    # parser.add_argument("-figure_show", action="store_true")
    parser.add_argument("-figure_show", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    results_dir = create_results_dir(
        "../results", f"{args.experiment_name}_{args.fusion_model}"
    )

    pprint_block("Dataset", color="green")
    np.random.seed(0)
    tf.random.set_seed(0)

    t_field = "t"
    a_field = "a"
    b_field = "b"

    t_a_field, t_b_field = t_field + "_a", t_field + "_b"

    # Generate Brownian motion signal
    signal_generator = SignalGenerator(add_degradation=False)

    t_a = signal_generator.t[signal_generator.t_a_indices]
    t_b = signal_generator.t[signal_generator.t_b_indices]
    a = signal_generator.a[signal_generator.t_a_indices]
    b = signal_generator.b[signal_generator.t_b_indices]

    pprint("Signal", level=0)
    pprint("- " + t_a_field, t_a.shape, level=1)
    pprint("- " + a_field, a.shape, level=1)

    pprint("Signal", level=0)
    pprint("- " + t_b_field, t_b.shape, level=1)
    pprint("- " + b_field, b.shape, level=1)

    plot_signals(
        [
            (t_a, a, r"$a$", False),
            (t_b, b, r"$b$", False),
            (signal_generator.t, signal_generator.s, r"$s$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        show=args.figure_show,
    )

    pprint_block("Data Fusion", color="green")
    gpf.config.set_default_float(np.float64)
    np.random.seed(0)
    tf.random.set_seed(0)

    labels, t_labels = build_labels([t_a, t_b])
    s = np.reshape(np.hstack((a, b)), newshape=(-1, 1))
    t = np.hstack((t_a, t_b))
    t = concatenate_labels(t, t_labels)
    t, s = sort_inputs(t, s, sort_axis=0)

    t_out = signal_generator.t
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
        pprint_block("Training", level=2, color="yellow")
        fusion_model.fit(windows=local_windows, max_iter=args.max_iter)

        # Predict
        pprint_block("Inference", level=2, color="yellow")
        s_out_mean, s_out_std = fusion_model(t_out, verbose=True)
    else:
        fusion_model = tsipy.fusion.SVGPModel(
            kernel=kernel,
            num_inducing_pts=args.num_inducing_pts,
            normalization=args.normalization,
            clipping=args.clipping,
        )

        # Train
        pprint_block("Training", level=2, color="yellow")
        fusion_model.fit(t, s, max_iter=args.max_iter, x_val=t_out, n_evals=5)

        # Predict
        pprint_block("Inference", level=2, color="yellow")
        s_out_mean, s_out_std = fusion_model(t_out)

    pprint_block("Results", color="green")
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
    ax.plot(signal_generator.t, signal_generator.s, label=r"$s$")
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
