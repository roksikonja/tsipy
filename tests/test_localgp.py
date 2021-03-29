import gpflow as gpf
import numpy as np

import tsipy.fusion
import tsipy.fusion
from tests.utils import check_array_approximate
from tsipy.correction import SignalGenerator
from tsipy.fusion.utils import build_labels, build_output_labels, concatenate_labels
from tsipy.utils import pprint_block, sort_inputs
from tsipy_utils.visualizer import plot_signals, plot_signals_and_confidence


def test_equivalence_localgp_and_svgp(show=False, verbose=False):
    if verbose:
        pprint_block("Equivalence between SVGP and LocalGP", color="green")

    signal_generator = SignalGenerator(
        length=10_000, add_degradation=False, random_seed=1
    )

    x_a = signal_generator.t[signal_generator.t_a_indices]
    x_b = signal_generator.t[signal_generator.t_b_indices]
    y_a = signal_generator.a[signal_generator.t_a_indices]
    y_b = signal_generator.b[signal_generator.t_b_indices]

    plot_signals(
        [
            (x_a, y_a, r"$a$", False),
            (x_b, y_b, r"$b$", False),
        ],
        show=show,
    )

    labels, x_labels = build_labels([x_a, x_b])
    y = np.reshape(np.hstack((y_a, y_b)), newshape=(-1, 1))
    x = np.hstack((x_a, x_b))
    x = concatenate_labels(x, x_labels)
    x, y = sort_inputs(x, y, sort_axis=0)

    x_out = signal_generator.t
    x_out_labels = build_output_labels(x_out)
    x_out = concatenate_labels(x_out, x_out_labels)

    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(labels=labels, active_dims=[1])
    kernel = matern_kernel + white_kernel

    # SVGP
    pprint_block("SVGP", color="yellow", level=1)
    fusion_model_svgp = tsipy.fusion.SVGPModel(
        kernel=kernel,
        num_inducing_pts=200,
        normalization=False,
        clipping=False,
    )
    fusion_model_svgp.fit(x, y, max_iter=2000, random_seed=1, verbose=verbose)
    y_out_mean_svgp, y_out_std_svgp = fusion_model_svgp(x_out)

    # Local GP with 1 window
    pprint_block("LocalGP", color="yellow", level=1)
    local_model = tsipy.fusion.SVGPModel(
        kernel=kernel,
        num_inducing_pts=200,
    )
    local_windows = tsipy.fusion.local_gp.create_windows(
        x,
        y,
        pred_window=1.0,
        fit_window=1.0,
        verbose=verbose,
    )
    fusion_model_local = tsipy.fusion.local_gp.LocalGPModel(
        model=local_model,
        normalization=False,
        clipping=False,
    )
    fusion_model_local.fit(
        windows=local_windows, max_iter=2000, random_seed=1, verbose=verbose
    )
    y_out_mean_local, y_out_std_local = fusion_model_local(x_out)

    pprint_block("Check results and visualize", color="yellow", level=1)
    plot_signals(
        [
            (x_out[:, 0], y_out_mean_local + 1, "LocalGP +1", False),
            (x_out[:, 0], y_out_mean_svgp - 1, "SVGP -1", False),
            (signal_generator.t, signal_generator.s, "GT", False),
        ],
        legend="upper left",
        show=show,
    )

    check_array_approximate(y_out_mean_local, y_out_mean_svgp, tolerance=0.2)
    check_array_approximate(y_out_std_local, y_out_std_svgp, tolerance=0.2)


def test_visualize_localgp(show=False, verbose=False):
    if verbose:
        pprint_block("V", color="green")

    signal_generator = SignalGenerator(
        length=10_000, add_degradation=False, random_seed=1
    )

    x_a = signal_generator.t[signal_generator.t_a_indices]
    x_b = signal_generator.t[signal_generator.t_b_indices]
    y_a = signal_generator.a[signal_generator.t_a_indices]
    y_b = signal_generator.b[signal_generator.t_b_indices]

    labels, x_labels = build_labels([x_a, x_b])
    y = np.reshape(np.hstack((y_a, y_b)), newshape=(-1, 1))
    x = np.hstack((x_a, x_b))
    x = concatenate_labels(x, x_labels)
    x, y = sort_inputs(x, y, sort_axis=0)

    _, ax_ful = plot_signals(
        [
            (x_a, y_a, r"$a$", False),
            (x_b, y_b, r"$b$", False),
        ],
        show=show,
    )

    x_out = signal_generator.t
    x_out_labels = build_output_labels(x_out)
    x_out = concatenate_labels(x_out, x_out_labels)

    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(labels=labels, active_dims=[1])
    kernel = matern_kernel + white_kernel

    # Local GP with 1 window
    local_model = tsipy.fusion.SVGPModel(
        kernel=kernel,
        num_inducing_pts=200,
    )
    local_windows = tsipy.fusion.local_gp.create_windows(
        x,
        y,
        pred_window=0.3,
        fit_window=0.4,
        verbose=verbose,
    )
    fusion_model_local = tsipy.fusion.local_gp.LocalGPModel(
        model=local_model,
        normalization=False,
        clipping=False,
    )
    pprint_block("Training", level=2)
    fusion_model_local.fit(
        windows=local_windows, max_iter=2000, random_seed=1, verbose=verbose
    )

    for window_id in range(len(local_windows)):
        y_out_mean_local, y_out_std_local = fusion_model_local.predict_window(
            x_out, window_id=window_id
        )

        window = local_windows[window_id]
        model = window.model
        x_inducing = model.x_inducing[:, 0]
        y_inducing = np.mean(y_out_mean_local) * np.ones_like(x_inducing)

        fig, ax = plot_signals_and_confidence(
            [(x_out[:, 0], y_out_mean_local, y_out_std_local, "LocalGP")],
        )
        ax.plot(signal_generator.t, signal_generator.s, label="$s$")
        ax.plot(
            x_inducing,
            y_inducing,
            "k|",
            mew=1,
            label="SVGP_INDUCING_POINTS",
        )
        ax.axvline(x=max(window.x_pred_start, x_out[0, 0]), color="tab:orange", ls="--")
        ax.axvline(x=min(window.x_pred_end, x_out[-1, 0]), color="tab:orange", ls="--")
        ax.axvline(x=window.x_pred_mid, color="k", ls="--")
        ax.axvline(x=max(window.x_fit_start, x_out[0, 0]), color="tab:orange")
        ax.axvline(x=min(window.x_fit_end, x_out[-1, 0]), color="tab:orange")
        ax.set_xlim(*ax_ful.get_xlim())
        ax.set_ylim(*ax_ful.get_ylim())
        if show:
            fig.show()


if __name__ == "__main__":
    test_equivalence_localgp_and_svgp(show=True, verbose=True)
    test_visualize_localgp(show=True, verbose=True)
