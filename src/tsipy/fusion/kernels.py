"""Module implements custom kernels for multi-sensor fusion."""
from typing import List, Optional, Tuple

import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive

__all__ = ["MultiWhiteKernel"]


class MultiWhiteKernel(gpf.kernels.Kernel):
    """Covariance function simulating Gaussian measurement noise.

    Each sensor has its own learnable noise variance parameter.
    Kernel returns a noise variance for a given sensor label (integer), i.e.
    if the n-th measurement was made by `i`-th instrument,
    it returns :math:`\\sigma_i^2` as given by:

    .. math::

        k(x_n, x_m) = \\delta(n, m) \\cdot \\sigma_i^2,

    where :math:`\\delta(\\cdot, \\cdot)` is the Kronecker delta function and
    :math:`\\sigma_i^2` is the variance of the `i`-th sensor measurement noise.

    Implementation is based on the following guidelines

        - `Kernel Design`_

        - `Active Dimensions`_.

    .. _Kernel Design:  \
    https://gpflow.readthedocs.io/en/master/notebooks/tailor/kernel_design.html

    .. _Active Dimensions: \
    https://gpflow.readthedocs.io/en/master/notebooks/advanced/kernels.html
    """

    def __init__(
        self,
        labels: Tuple[int, ...],
        variances: Optional[np.ndarray] = None,
        active_dims: List[int] = None,
    ):
        super().__init__(active_dims=active_dims)
        self.labels = np.array(labels)
        self.n = len(labels)

        if variances is None:
            variances = tf.ones((self.n,))

        self.variances = gpf.Parameter(
            variances, transform=positive(), dtype=gpf.default_float()
        )

    # noinspection PyPep8Naming
    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        # pylint: disable=C0103
        if X2 is None:
            diag = self.K_diag(X)
            return tf.linalg.diag(diag)

        shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], axis=0)
        return tf.zeros(shape, dtype=X.dtype)

    # noinspection PyPep8Naming
    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        # pylint: disable=C0103
        X = tf.squeeze(X)

        diag = tf.zeros((tf.shape(X)[0],), dtype=X.dtype)  # (None, 1)
        for i in range(self.n):
            mask = tf.cast(tf.equal(X, i + 1), dtype=X.dtype)  # (None, 1)
            mask_noise = tf.multiply(mask, self.variances[i])  # (None, 1)

            diag = diag + tf.squeeze(mask_noise)
        return diag
