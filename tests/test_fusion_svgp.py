import gpflow as gpf

import tsipy.fusion
from tests.utils import check_array_approximate
from tests.utils_fusion import load_data_with_labels, load_data_without_labels
from tsipy.utils import pprint_block, pprint
from tsipy_utils.visualizer import plot_signals


def test_svgp_convergence_with_labels(verbose: bool = True, show: bool = True) -> None:
    if verbose:
        pprint_block("Convergence of SVGP to ground truth with sensor labels")

    x_a, x_b, x, y_a, y_b, y, x_out, x_gt, y_gt = load_data_with_labels(random_seed=0)

    plot_signals(
        [
            (x_a, y_a, "$a$", {}),
            (x_b, y_b, "$b$", {}),
        ],
        show=show,
    )

    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(labels=(1, 2), active_dims=[1])
    kernel = matern_kernel + white_kernel

    # SVGP
    for normalization in [True, False]:
        for clipping in [True, False]:
            for num_inducing_pts, max_iter in [(200, 2000), (100, 1000), (250, 2000)]:
                if verbose:
                    pprint_block("Training SVGP", level=1)
                    pprint("normalization:", normalization)
                    pprint("clipping:", clipping)
                    pprint("num_inducing_pts:", num_inducing_pts)
                    pprint("max_iter:", max_iter)

                fusion_model = tsipy.fusion.SVGPModel(
                    kernel=kernel,
                    num_inducing_pts=num_inducing_pts,
                    normalization=normalization,
                    clipping=clipping,
                )

                fusion_model.fit(
                    x, y, max_iter=max_iter, random_seed=1, verbose=verbose
                )
                y_out_mean, y_out_std = fusion_model(x_out)

                plot_signals(
                    [
                        (
                            x_gt,
                            y_gt,
                            "GT",
                            {"c": "tab:red"},
                        ),
                        (x_out[:, 0], y_out_mean, "SVGP", {}),
                    ],
                    legend="upper left",
                    show=show,
                )

                check_array_approximate(y_gt, y_out_mean, tolerance=0.11)


def test_svgp_convergence_without_labels(
    verbose: bool = True, show: bool = True
) -> None:
    if verbose:
        pprint_block("Convergence of SVGP to ground truth without sensor labels")

    x_a, x_b, x, y_a, y_b, y, x_out, x_gt, y_gt = load_data_without_labels(
        random_seed=0
    )

    plot_signals(
        [
            (x_a, y_a, "$a$", {}),
            (x_b, y_b, "$b$", {}),
        ],
        show=show,
    )

    kernel = gpf.kernels.Matern12(active_dims=[0]) + gpf.kernels.White(active_dims=[0])

    # SVGP
    for normalization in [True, False]:
        for clipping in [True, False]:
            for num_inducing_pts, max_iter in [(200, 2000), (100, 1000), (250, 2000)]:
                if verbose:
                    pprint_block("Training SVGP", level=1)
                    pprint("normalization:", normalization)
                    pprint("clipping:", clipping)
                    pprint("num_inducing_pts:", num_inducing_pts)
                    pprint("max_iter:", max_iter)

                fusion_model = tsipy.fusion.SVGPModel(
                    kernel=kernel,
                    num_inducing_pts=num_inducing_pts,
                    normalization=normalization,
                    clipping=clipping,
                )

                fusion_model.fit(
                    x, y, max_iter=max_iter, random_seed=1, verbose=verbose
                )
                y_out_mean, y_out_std = fusion_model(x_out)

                plot_signals(
                    [
                        (
                            x_gt,
                            y_gt,
                            "GT",
                            {"c": "tab:red"},
                        ),
                        (x_out[:, 0], y_out_mean, "SVGP", {}),
                    ],
                    legend="upper left",
                    show=show,
                )

                check_array_approximate(y_gt, y_out_mean, tolerance=0.11)
