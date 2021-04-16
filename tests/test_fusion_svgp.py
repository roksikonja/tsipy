import gpflow as gpf

from tests.utils import check_array_approximate
from tests.utils_fusion import load_data_with_labels, load_data_without_labels
from tsipy.fusion import SVGPModel
from tsipy.fusion.kernels import MultiWhiteKernel
from tsipy.utils import pprint, pprint_block


def test_svgp_convergence_with_labels(verbose: bool = True) -> None:
    if verbose:
        pprint_block("Convergence of SVGP to ground truth with sensor labels")

    random_seed = 1
    x_a, x_b, x, y_a, y_b, y, x_out, x_gt, y_gt = load_data_with_labels(
        random_seed=random_seed
    )

    matern_kernel = gpf.kernels.Matern12(active_dims=[0])
    white_kernel = MultiWhiteKernel(labels=(1, 2), active_dims=[1])
    kernel = matern_kernel + white_kernel

    _run_svgp(x, y, y_gt, x_out, kernel, random_seed, verbose)


def test_svgp_convergence_without_labels(verbose: bool = True) -> None:
    if verbose:
        pprint_block("Convergence of SVGP to ground truth without sensor labels")

    random_seed = 1
    x_a, x_b, x, y_a, y_b, y, x_out, x_gt, y_gt = load_data_without_labels(
        random_seed=random_seed
    )

    kernel = gpf.kernels.Matern12(active_dims=[0]) + gpf.kernels.White(active_dims=[0])
    _run_svgp(x, y, y_gt, x_out, kernel, random_seed, verbose)


def _run_svgp(x, y, y_gt, x_out, kernel, random_seed, verbose):
    normalization = True
    clipping = True
    for num_inducing_pts, max_iter in [(200, 2000), (100, 1000), (250, 2000)]:
        if verbose:
            pprint_block("Training SVGP", level=1)
            pprint("normalization:", normalization)
            pprint("clipping:", clipping)
            pprint("num_inducing_pts:", num_inducing_pts)
            pprint("max_iter:", max_iter)

        fusion_model = SVGPModel(
            kernel=kernel,
            num_inducing_pts=num_inducing_pts,
            normalization=normalization,
            clipping=clipping,
        )

        fusion_model.fit(
            x, y, max_iter=max_iter, random_seed=random_seed, verbose=verbose
        )
        y_out_mean, y_out_std = fusion_model(x_out)

        check_array_approximate(y_gt, y_out_mean, tolerance=0.11)
