import argparse
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf

import tsipy.fusion
from tsipy.correction.generator import SignalGenerator
from tsipy.fusion.utils import (
    build_sensor_labels,
    build_output_labels,
    concatenate_labels,
)
from tsipy.utils import pprint, pprint_block
from utils import Constants as Const
from utils.data import create_results_dir
from utils.visualizer import plot_signals_and_confidence, plot_signals


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="demo_fusion", type=str)

    # Fusion Model
    parser.add_argument("--fusion_model", default="svgp", type=str)

    # SVGP
    parser.add_argument("--num_inducing_pts", default=1000, type=int)
    parser.add_argument("--max_iter", default=8000, type=int)

    # Local GP
    parser.add_argument("--pred_window", default=0.2, type=float)
    parser.add_argument("--fit_window", default=0.5, type=float)

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    results_dir = create_results_dir(
        "../results", f"{args.experiment_name}_{args.fusion_model}"
    )

    """
        Dataset
    """
    pprint_block("Dataset")
    np.random.seed(Const.RANDOM_SEED)
    tf.random.set_seed(Const.RANDOM_SEED)

    t_field = "t"
    a_field = "a"
    b_field = "b"

    t_a_field, t_b_field = t_field + "_a", t_field + "_b"

    # Generate Brownian motion signal
    signal_generator = SignalGenerator(add_degradation=False)
    data = signal_generator.data

    t_a = signal_generator.t[signal_generator.t_a_indices]
    t_b = signal_generator.t[signal_generator.t_b_indices]
    a = signal_generator.a[signal_generator.t_a_indices]
    b = signal_generator.b[signal_generator.t_b_indices]

    print(data, "\n")
    pprint("- data", data.shape)
    pprint("- " + t_a_field, t_a.shape)
    pprint("- " + t_b_field, t_b.shape)
    pprint("- " + a_field, a.shape, np.sum(~np.isnan(a)))
    pprint("- " + b_field, b.shape, np.sum(~np.isnan(b)))

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

    """
        Data fusion
    """
    pprint_block("Data Fusion")
    gpf.config.set_default_float(np.float64)
    np.random.seed(Const.RANDOM_SEED)
    tf.random.set_seed(Const.RANDOM_SEED)

    labels, t_labels = build_sensor_labels((t_a, t_b))
    s = np.hstack((a, b))
    t = np.hstack((t_a, t_b))
    t = concatenate_labels(t, t_labels, sort_axis=0)

    t_out = signal_generator.t
    t_out_labels = build_output_labels(t_out)
    t_out = concatenate_labels(t_out, t_out_labels)

    pprint("- labels", labels)
    pprint("- t_labels", t_labels.shape)
    pprint("- t", t.shape)
    pprint("- s", s.shape)
    pprint("- t_out_labels", t_out_labels.shape)
    pprint("- t_out", t_out.shape)

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

        fusion_model = tsipy.fusion.local_gp.LocalGPModel(
            model=local_model, windows=local_windows
        )

        # Train
        pprint_block("Training", level=2)
        fusion_model.fit(max_iter=args.max_iter, verbose=True)

        # Predict
        pprint_block("Inference", level=2)
        s_out_mean, s_out_std = fusion_model(t_out, verbose=True)
    else:
        fusion_model = tsipy.fusion.SVGPModel(
            kernel=kernel, num_inducing_pts=args.num_inducing_pts
        )

        # Train
        pprint_block("Training", level=2)
        fusion_model.fit(t, s, max_iter=args.max_iter, x_val=t_out, n_evals=5)

        # Predict
        pprint_block("Inference", level=2)
        s_out_mean, s_out_std = fusion_model(t_out)

    """
        Composite
    """
    pprint_block("Results")
    t_out = t_out[:, 0]

    pprint("    - t_out", t_out.shape)
    pprint("    - s_out_mean", s_out_mean.shape)
    pprint("    - s_out_std", s_out_std.shape)

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
    )
    ax.scatter(
        t_a,
        a,
        label="$a$",
        s=Const.MARKER_SIZE,
    )
    ax.scatter(
        t_b,
        b,
        label="$b$",
        s=Const.MARKER_SIZE,
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
