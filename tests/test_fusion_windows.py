from typing import Dict, List, Tuple

import numpy as np

import tsipy.fusion
from tests.utils import check_window_ranges, check_array_equal
from tests.utils_fusion import load_data_with_labels
from tsipy.fusion.local_gp import create_windows
from tsipy.utils import pprint_block
from tsipy_utils.visualizer import plot_signals


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


def test_visualize_windows_one(verbose: bool = False, show: bool = False) -> None:
    if verbose:
        pprint_block("Visualize windows")

    random_seed = 1
    x_a, x_b, x, y_a, y_b, y, x_out, x_gt, y_gt = load_data_with_labels(
        random_seed=random_seed
    )

    _, ax_ful = plot_signals(
        [
            (x_a[:, 0], y_a, "$a$", {}),
            (x_b[:, 0], y_b, "$b$", {}),
        ],
        show=show,
    )

    for pred_window_width, fit_window_width in [(0.3, 0.4), (1.0, 1.0)]:
        windows = tsipy.fusion.local_gp.create_windows(
            x,
            y,
            pred_window_width=pred_window_width,
            fit_window_width=fit_window_width,
            verbose=verbose,
        )
        for window in windows:
            x_window = window.x
            y_window = window.y

            signal_fiveplets: List[Tuple[np.ndarray, np.ndarray, str, Dict]] = []
            for label, label_str in zip([1, 2], ["a", "b"]):
                label_indices = np.equal(x_window[:, 1], label)
                signal_fiveplets.append(
                    (
                        x_window[label_indices, 0],
                        y_window[label_indices, 0],
                        label_str,
                        {},
                    )
                )
            signal_fiveplets.append((x_gt, y_gt, "GT", {}))

            fig, ax = plot_signals(
                signal_fiveplets,
                legend="upper left",
            )
            ax.axvline(x=window.x_pred_start, color="k")
            ax.axvline(x=window.x_pred_end, color="k")
            ax.axvline(x=window.x_pred_mid, color="k", ls="--")
            ax.set_xlim(*ax_ful.get_xlim())
            ax.set_ylim(*ax_ful.get_ylim())
            if show:
                fig.show()
