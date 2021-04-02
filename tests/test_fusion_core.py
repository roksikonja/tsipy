import numpy as np

from tsipy.fusion.core import NormalizeAndClip


def test_normalize_and_clip():
    """Tests that accessing non-initialized attribute raises an error."""
    nc = NormalizeAndClip(normalization=False, clipping=False)

    try:
        print(nc.y_shift)
    except AssertionError:
        pass

    x = np.reshape(np.array([0.0, 1.0, 2.0]), newshape=(-1, 1))
    y = np.reshape(np.array([0.0, 1.0, 2.0]), newshape=(-1, 1))

    nc.compute_normalization_values(x, y)
    nc.normalize_and_clip(x, y)

    assert nc.x_shift is not None and nc.x_shift == 0.0
    assert nc.x_scale is not None and nc.x_scale == 1.0
    assert nc.y_shift is not None and nc.y_shift == 0.0
    assert nc.y_scale is not None and nc.y_scale == 1.0

    # Test inplace normalization
    x_norm = nc.normalize_x(x, x_shift=2.0)
    y_norm = nc.normalize_y(y, y_shift=2.0)
    assert id(x_norm) != id(x), "x is modified inplace."
    assert id(y_norm) != id(y), "y is modified inplace."

    # Check returned values are 2D
    assert len(x_norm.shape) == 2, "Input x_norm with shape {} is not 2D.".format(
        x_norm.shape
    )
    assert len(y_norm.shape) == 2, "Input y_norm with shape {} is not 2D.".format(
        y_norm.shape
    )

    try:
        nc.compute_normalization_values(x, y)
        assert False, "Attempted to re-compute normalization values."
    except ValueError:
        assert True
