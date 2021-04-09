import numpy as np

from tests.utils import check_array_equal, check_window_ranges
from tsipy.fusion.local_gp import create_windows
from tsipy.utils import pprint_block


def test_ranges_one(verbose: bool = False) -> None:
    if verbose:
        pprint_block("Test", "1", color="green")

    x = np.reshape(np.linspace(0.0, 1.0, 1000 + 1), newshape=(-1, 1))
    y = np.random.randn(*x.shape)

    windows = create_windows(
        x, y, pred_window_width=0.2, fit_window_width=0.4, verbose=verbose
    )
    check_window_ranges(
        windows,
        x_pred_starts=np.array([-np.infty, 0.2, 0.4, 0.6, 0.8]),
        x_pred_ends=np.array([0.2, 0.4, 0.6, 0.8, np.infty]),
        x_fit_starts=np.array([0.0, 0.1, 0.3, 0.5, 0.7]),
        x_fit_ends=np.array([0.3, 0.5, 0.7, 0.9, 1.0]),
    )


def test_ranges_two(verbose: bool = False) -> None:
    if verbose:
        pprint_block("Test", "2", color="green")

    x = np.reshape(np.linspace(0.3, 1.0, 1000 + 1), newshape=(-1, 1))
    y = np.random.randn(*x.shape)

    windows = create_windows(
        x, y, pred_window_width=0.3, fit_window_width=0.4, verbose=verbose
    )
    check_window_ranges(
        windows,
        x_pred_starts=np.array([-np.infty, 0.6, 0.9]),
        x_pred_ends=np.array([0.6, 0.9, np.infty]),
        x_fit_starts=np.array([0.3, 0.55, 0.85]),
        x_fit_ends=np.array([0.65, 0.95, 1.00]),
    )


def test_ranges_three(verbose: bool = False) -> None:
    if verbose:
        pprint_block("Test", "3", color="green")

    x = np.reshape(np.linspace(-0.3, 0.7, 2000 + 1), newshape=(-1, 1))
    y = np.random.randn(*x.shape)

    windows = create_windows(
        x, y, pred_window_width=1.0, fit_window_width=1.0, verbose=verbose
    )
    check_window_ranges(
        windows,
        x_pred_starts=np.array([-np.infty]),
        x_pred_ends=np.array([np.infty]),
        x_fit_starts=np.array([-0.3]),
        x_fit_ends=np.array([0.7]),
    )


def test_windows_gather_data(verbose: bool = False) -> None:
    if verbose:
        pprint_block("Test", "3", color="green")

    for i in range(10):
        if verbose:
            pprint_block("Random seed", str(i), level=1, color="yellow")
        np.random.seed(i)

        x = np.reshape(np.linspace(-0.3, 3.0, 5000 + 1), newshape=(-1, 1))
        y = np.random.randn(*x.shape)

        windows = create_windows(
            x,
            y,
            pred_window_width=np.random.uniform(0.1, 0.4),
            fit_window_width=np.random.uniform(0.4, 1.0),
            verbose=verbose,
        )

        gather_x, gather_y = windows.gather_data()

        check_array_equal(gather_x, x)
        check_array_equal(gather_y, y)
