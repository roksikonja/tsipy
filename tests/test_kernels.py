import numpy as np
import tensorflow as tf
from gpflow.kernels import Linear
from gpflow.utilities import print_summary

from tsipy.fusion.kernels import MultiWhiteKernel
from tsipy.fusion.utils import build_labels


def test_kernel():
    """
    Test implementation of the custom white kernel.
    """

    np.random.seed(0)

    x_a = np.random.randint(0, 5, 3)
    x_b = np.random.randint(0, 5, 2)
    x_c = np.random.randint(0, 5, 1)

    x = np.hstack((x_a, x_b, x_c))
    labels, x_labels = build_labels([x_a, x_b, x_c])
    x = np.vstack((x, x_labels)).T

    x = tf.convert_to_tensor(x, dtype=tf.float64)
    x2 = x[:-1, :]

    # White kernel
    noise_variances = tf.convert_to_tensor(0.1 * labels, dtype=tf.float64)

    linear_kernel = Linear(active_dims=[0])
    multi_kernel = MultiWhiteKernel(
        labels=labels, variances=noise_variances, active_dims=[1]
    )

    n = 3
    variances = tf.zeros((x.shape[0],), dtype=tf.float64)
    for i in range(n):
        mask = tf.cast(tf.equal(x[:, -1], i + 1), dtype=tf.float64)
        mask_variances = mask * noise_variances[i]
        variances = variances + mask_variances

    # Test
    print_summary(linear_kernel)
    print_summary(multi_kernel)

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
