import gpflow as gpf
import numpy as np

from tests.utils import check_array_approximate
from tests.utils_fusion import load_data_with_labels, load_data_without_labels
from tsipy.fusion import SVGPModel
from tsipy.fusion.kernels import MultiWhiteKernel


def test_svgp_convergence_with_labels(verbose: bool = False) -> None:
    """Tests convergence of SVGP fusion model to ground truth.

    A multi-white kernel is used to simulate measurement noise with added
    sensor labels.
    """
    random_seed = 1
    _, _, x, _, _, y, x_out, _, y_gt = load_data_with_labels(random_seed=random_seed)

    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = MultiWhiteKernel(labels=(1, 2), active_dims=[1])
    kernel = matern_kernel + white_kernel

    _run_svgp(
        x=x,
        y=y,
        y_gt=y_gt,
        x_out=x_out,
        kernel=kernel,
        random_seed=random_seed,
        verbose=verbose,
    )


def test_svgp_convergence_without_labels(verbose: bool = False) -> None:
    """Tests convergence of SVGP fusion model to ground truth.

    A standard white kernel is used to simulate measurement noise.
    """
    random_seed = 1
    _, _, x, _, _, y, x_out, _, y_gt = load_data_without_labels(random_seed=random_seed)

    kernel = gpf.kernels.Matern12(active_dims=[0]) + gpf.kernels.White(active_dims=[0])
    _run_svgp(
        x=x,
        y=y,
        y_gt=y_gt,
        x_out=x_out,
        kernel=kernel,
        random_seed=random_seed,
        verbose=verbose,
    )


def _run_svgp(
    x: np.ndarray,
    y: np.ndarray,
    y_gt: np.ndarray,
    x_out: np.ndarray,
    kernel: gpf.kernels.Kernel,
    random_seed: int,
    verbose: bool,
    tolerance: float = 0.11,
):
    """Runs SVGP training and checks convergence."""
    normalization = True
    clipping = True
    for num_inducing_pts, max_iter in [(200, 2000), (100, 1000), (250, 2000)]:
        fusion_model = SVGPModel(
            kernel=kernel,
            num_inducing_pts=num_inducing_pts,
            normalization=normalization,
            clipping=clipping,
        )

        fusion_model.fit(
            x, y, max_iter=max_iter, random_seed=random_seed, verbose=verbose
        )
        y_out_mean, _ = fusion_model(x_out)

        check_array_approximate(y_gt, y_out_mean, tolerance=tolerance)
