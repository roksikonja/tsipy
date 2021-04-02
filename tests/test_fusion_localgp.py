import gpflow as gpf
import numpy as np
import tsipy.fusion
from tests.utils import check_array_approximate
from tests.utils_fusion import load_data_with_labels
from tsipy.utils import pprint, pprint_block
from tsipy_utils.visualizer import plot_signals, plot_signals_and_confidence


def test_localgp_convergence_with_labels(
    show: bool = True, verbose: bool = True
) -> None:
    if verbose:
        pprint_block("Convergence of LocalGP to ground truth with sensor labels")

    random_seed = 1
    x_a, x_b, x, y_a, y_b, y, x_out, x_gt, y_gt = load_data_with_labels(
        random_seed=random_seed
    )

    plot_signals(
        [
            (x_a[:, 0], y_a, "$a$", {}),
            (x_b[:, 0], y_b, "$b$", {}),
        ],
        show=show,
    )

    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(labels=(1, 2), active_dims=[1])
    kernel = matern_kernel + white_kernel

    for normalization in [True, False]:
        for clipping in [True, False]:
            for num_inducing_pts, max_iter in [
                (200, 2000),
                (100, 1000),
                (250, 2000),
            ]:
                if verbose:
                    pprint_block("Training SVGP", level=1)
                    pprint("normalization:", normalization)
                    pprint("clipping:", clipping)
                    pprint("num_inducing_pts:", num_inducing_pts)
                    pprint("max_iter:", max_iter)

                local_model = tsipy.fusion.SVGPModel(
                    kernel=kernel,
                    num_inducing_pts=num_inducing_pts,
                    normalization=True,
                    clipping=True,
                )
                fusion_model = tsipy.fusion.local_gp.LocalGPModel(
                    model=local_model,
                    pred_window_width=1.0,
                    fit_window_width=1.0,
                    normalization=normalization,
                    clipping=clipping,
                )
                fusion_model.fit(
                    x, y, max_iter=max_iter, random_seed=1, verbose=verbose
                )
                y_out_mean, y_out_std = fusion_model(x_out)

                plot_signals(
                    [
                        (x_out[:, 0], y_out_mean, "LocalGP", {}),
                        (x_gt, y_gt, "GT", {}),
                    ],
                    legend="upper left",
                    show=show,
                )

                check_array_approximate(y_gt, y_out_mean, tolerance=0.05)


def test_visualize_localgp(show: bool = True, verbose: bool = True) -> None:
    if verbose:
        pprint_block(
            "Visualize windows",
        )

    random_seed = 1
    x_a, x_b, x, y_a, y_b, y, x_out, x_gt, y_gt = load_data_with_labels(
        random_seed=random_seed
    )

    _, ax_ful = plot_signals(
        [
            (x_a[:, 0], y_a, r"$a$", {}),
            (x_b[:, 0], y_b, r"$b$", {}),
        ],
        show=show,
    )

    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(labels=(1, 2), active_dims=[1])
    kernel = matern_kernel + white_kernel

    for pred_window_width, fit_window_width in [(0.2, 0.6)]:
        for normalization, clipping in [(False, False), (True, True)]:
            local_model = tsipy.fusion.SVGPModel(
                kernel=kernel,
                num_inducing_pts=50,
                normalization=True,
                clipping=True,
            )
            fusion_model = tsipy.fusion.local_gp.LocalGPModel(
                model=local_model,
                pred_window_width=pred_window_width,
                fit_window_width=fit_window_width,
                normalization=normalization,
                clipping=clipping,
            )
            fusion_model.fit(
                x, y, max_iter=1000, random_seed=random_seed, verbose=verbose
            )

            y_out_mean, y_out_std = fusion_model(x_out, verbose=verbose)
            for window_id, window in enumerate(fusion_model.windows):
                y_out_mean_window, y_out_std_window = fusion_model.predict_window(
                    x_out, window_id=window_id, verbose=verbose
                )

                fig, ax = plot_signals_and_confidence(
                    [
                        (x_out[:, 0], y_out_mean_window, y_out_std_window, "LocalGP-W"),
                        (x_out[:, 0], y_out_mean, y_out_std, "LocalGP"),
                    ],
                )
                ax.plot(x_gt, y_gt, label="$s$")

                ax.axvline(
                    x=max(window.x_pred_start, x_out[0, 0]), color="tab:orange", ls="--"
                )
                ax.axvline(
                    x=min(window.x_pred_end, x_out[-1, 0]), color="tab:orange", ls="--"
                )
                ax.axvline(x=window.x_pred_mid, color="k", ls="--")
                ax.axvline(x=max(window.x_fit_start, x_out[0, 0]), color="tab:orange")
                ax.axvline(x=min(window.x_fit_end, x_out[-1, 0]), color="tab:orange")
                ax.set_xlim(*ax_ful.get_xlim())
                ax.set_ylim(*ax_ful.get_ylim())

                if window.model.x_inducing is not None:
                    x_inducing = window.model.x_inducing
                    x_inducing = window.model._nc.denormalize_x(x_inducing)
                    x_inducing = fusion_model._nc.denormalize_x(x_inducing)
                    y_inducing = np.mean(y_out_mean) * np.ones_like(x_inducing)

                    ax.plot(
                        x_inducing[:, 0],
                        y_inducing,
                        "k|",
                        mew=1,
                        label="SVGP_INDUCING_POINTS",
                    )

                fig.tight_layout()
                if show:
                    fig.show()
            for window_id, window in enumerate(fusion_model.windows):
                print(str(window) + "\n")
                print(window.model.x_inducing[0, :], window.model.x_inducing[-1, :])
