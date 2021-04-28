import gpflow as gpf

from tests.utils import check_array_approximate
from tests.utils_fusion import load_data_with_labels
from tsipy.fusion import LocalGPModel, SVGPModel
from tsipy.fusion.kernels import MultiWhiteKernel
from tsipy.utils import plot_signals


def test_localgp_convergence_with_labels(
    show: bool = False, verbose: bool = False
) -> None:
    """Tests convergence of LocalGP.

    Arguments `show` and `verbose` are added for debugging purposes.
    """

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
    white_kernel = MultiWhiteKernel(labels=(1, 2), active_dims=[1])
    kernel = matern_kernel + white_kernel

    for normalization in [True, False]:
        for clipping in [True, False]:
            for num_inducing_pts, max_iter in [
                (200, 2000),
                (100, 1000),
                (250, 2000),
            ]:

                local_model = SVGPModel(
                    kernel=kernel,
                    num_inducing_pts=num_inducing_pts,
                    normalization=True,
                    clipping=True,
                )
                fusion_model = LocalGPModel(
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
