import numpy as np

from tsipy.fusion import Windows


def check_window_ranges(
    windows: Windows,
    x_pred_starts: np.ndarray,
    x_pred_ends: np.ndarray,
    x_fit_starts: np.ndarray,
    x_fit_ends: np.ndarray,
) -> None:
    """Checks if input windows have approximately equal window bounds.

    Checks prediction and training bounds.
    """
    for i, window in enumerate(windows):
        if np.abs(x_pred_starts[i]) == np.infty:
            assert x_pred_starts[i] == window.x_pred_start
        else:
            assert np.abs(x_pred_starts[i] - window.x_pred_start) < 1e-3

        if np.abs(x_pred_ends[i]) == np.infty:
            assert x_pred_ends[i] == window.x_pred_end
        else:
            assert np.abs(x_pred_ends[i] - window.x_pred_end) < 1e-3

        assert np.abs(x_fit_starts[i] - window.x_fit_start) < 1e-3, "{} != {}".format(
            x_fit_starts[i], window.x_fit_start
        )
        assert np.abs(x_fit_ends[i] - window.x_fit_end) < 1e-3, "{} != {}".format(
            x_fit_ends[i], window.x_fit_end
        )


def check_array_equal(a: np.ndarray, b: np.ndarray) -> None:
    """Checks if both arrays are equal."""
    assert np.array_equal(a, b), "Array a not equal to array b."


def check_array_approximate(
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float = 1e-3,
) -> None:
    """Checks if both arrays are approximately equal."""
    check_array_shape(a, b)

    norm_a = np.linalg.norm(a)
    error = np.linalg.norm(a - b) / norm_a
    assert (
        error < tolerance
    ), "Array a not equal to array b. ({:.6f} < {}, error < tolerance)".format(
        error, tolerance
    )


def check_array_shape(a: np.ndarray, b: np.ndarray) -> None:
    """Checks if both arrays have identical shapes."""
    assert np.array_equal(
        a.shape, b.shape
    ), "Array a {} and b {} shape mismatched.".format(a.shape, b.shape)


def check_array_1d(a: np.ndarray) -> None:
    """Checks if array is one dimensional."""
    assert len(a.shape) == 1, "Array a with shape {} is not 1D.".format(a.shape)
    assert a.ndim == 1, "Array a with shape {} is not 1D.".format(a.shape)


def check_array_2d(a: np.ndarray) -> None:
    """Checks if array is one dimensional."""
    assert len(a.shape) == 2, "Array a with shape {} is not 2D.".format(a.shape)
    assert a.ndim == 2, "Array a with shape {} is not 2D.".format(a.shape)
