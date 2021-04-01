from typing import List, Optional, Tuple

import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive


class MultiWhiteKernel(gpf.kernels.Kernel):
    """
    Implementation guidelines.
    https://gpflow.readthedocs.io/en/master/notebooks/tailor/kernel_design.html?highlight=kernel
    https://gpflow.readthedocs.io/en/master/notebooks/advanced/kernels.html#Specify-active-dimensions
    """

    def __init__(
        self,
        labels: Tuple[int, ...],
        variances: Optional[np.ndarray] = None,
        active_dims: List[int] = None,
    ):
        """
        The multiple White kernel: this kernel produces 'white noise'. The kernel equation is
            k(x_n, x_m) = δ(n, m) σ_i²
            if n-th measurement was made by the i-th instrument.
        where:
            δ(.,.) is the Kronecker delta,
            σ_i²  is the noise variance parameter of i-th instrument.
        """
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
        if X2 is None:
            diag = self.K_diag(X)
            return tf.linalg.diag(diag)
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], axis=0)
            return tf.zeros(shape, dtype=X.dtype)

    # noinspection PyPep8Naming
    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        X = tf.squeeze(X)

        diag = tf.zeros((tf.shape(X)[0],), dtype=X.dtype)  # (None, 1)
        for i in range(self.n):
            mask = tf.cast(tf.equal(X, i + 1), dtype=X.dtype)  # (None, 1)
            mask_noise = tf.multiply(mask, self.variances[i])  # (None, 1)

            diag = diag + tf.squeeze(mask_noise)
        return diag
