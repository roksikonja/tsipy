import numpy as np
import tensorflow as tf
from gpflow.kernels import Linear

from tsipy.fusion.kernels import MultiWhiteKernel
from tsipy.fusion.utils import build_and_concat_label_mask


def test_kernel() -> None:
    """Tests implementation of the custom white kernel.

    It checks output tensor shapes and values.
    """
    np.random.seed(0)

    x_a = np.random.randint(0, 5, 3)
    x_b = np.random.randint(0, 5, 2)
    x_c = np.random.randint(0, 5, 1)

    x_a = build_and_concat_label_mask(x_a, label=1)
    x_b = build_and_concat_label_mask(x_b, label=2)
    x_c = build_and_concat_label_mask(x_c, label=3)

    # Concatenate signals and sort by x[:, 0]
    x = np.vstack((x_a, x_b, x_c))

    x = tf.convert_to_tensor(x, dtype=tf.float64)
    x2 = x[:-1, :]

    # White kernel
    noise_variances = tf.convert_to_tensor(
        0.1 * np.array([1.0, 2.0, 3.0]), dtype=tf.float64
    )

    linear_kernel = Linear(active_dims=[0])
    multi_kernel = MultiWhiteKernel(
        labels=(1, 2, 3), variances=noise_variances, active_dims=[1]
    )

    n = 3
    variances = tf.zeros((x.shape[0],), dtype=tf.float64)
    for i in range(n):
        mask = tf.cast(tf.equal(x[:, -1], i + 1), tf.float64)
        mask_variances = mask * noise_variances[i]
        variances = variances + mask_variances

    # Test dimensions
    assert np.array_equal(linear_kernel(x).numpy().shape, multi_kernel(x).numpy().shape)
    assert np.array_equal(
        linear_kernel(x2).numpy().shape, multi_kernel(x2).numpy().shape
    )

    assert np.array_equal(
        linear_kernel(x, x2).numpy().shape, multi_kernel(x, x2).numpy().shape
    )
    assert np.array_equal(
        linear_kernel(x2, x).numpy().shape, multi_kernel(x2, x).numpy().shape
    )

    # Test variances
    assert np.linalg.norm(multi_kernel(x).numpy() - np.diag(variances)) < 1e-9
