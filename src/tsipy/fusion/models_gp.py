from typing import Optional, Tuple, List

import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow.utilities.utilities import tabulate_module_summary, default_summary_fmt

from ..fusion.core import NormalizeAndClip, FusionModel
from ..utils import (
    find_nearest_indices,
    pprint,
)


class SVGPModel(FusionModel):
    def __init__(
        self,
        kernel: gpf.kernels.Kernel,
        num_inducing_pts: int = 1000,
        inducing_trainable: bool = False,
        normalization: bool = True,
        clipping: bool = True,
    ) -> None:
        # Helper object for data normalization and clipping
        self._nc = NormalizeAndClip(normalization=normalization, clipping=clipping)

        self._model: Optional[gpf.models.GPModel] = None
        self._kernel = kernel
        self.x_inducing: Optional[np.ndarray] = None
        self.num_inducing_pts = num_inducing_pts
        self.inducing_trainable = inducing_trainable

        # Training history
        self.iter_elbo: Optional[np.ndarray] = None
        self.t_prior: Optional[np.ndarray] = None
        self.t_posterior: Optional[np.ndarray] = None
        self.history: Optional[List] = None

    def __str__(self) -> str:
        return tabulate_module_summary(self._model, default_summary_fmt())

    def __call__(
        self, x: np.ndarray, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Equivalent to :member:`predict` method."""
        y_mean, y_std = self.predict(x)
        return y_mean, y_std

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self._model is not None, "Model is not initialized."

        x_norm: np.ndarray = self._nc.normalize_x(x)
        y_mean, y_var = self._model.predict_y(tf.convert_to_tensor(x_norm))

        y_mean = y_mean.numpy()
        y_std = np.sqrt(y_var.numpy())

        y_mean = self._nc.denormalize_y(y_mean)
        y_std = self._nc.denormalize_y(y_std, y_shift=0.0)
        return y_mean.ravel(), y_std.ravel()

    def _build(
        self,
        x: np.ndarray,
        x_inducing: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initializes inducing variables and builds the SVGP model."""
        self.set_random_seed(random_seed)

        # Inducing variables
        if x_inducing is None:
            x_uniform = np.linspace(
                np.min(x[:, 0]), np.max(x[:, 0]), self.num_inducing_pts + 1
            )
            x_uniform_indices = find_nearest_indices(x[:, 0], x_uniform)
            x_inducing = np.copy(x[x_uniform_indices, :])
            self.x_inducing = x_inducing
        else:
            assert (
                len(x_inducing.shape) == 2
            ), "x_inducing with shape {} is not 2D.".format(x_inducing.shape)
            self.x_inducing = x_inducing

        self._model = gpf.models.SVGP(
            self._kernel,
            gpf.likelihoods.Gaussian(),
            self.x_inducing,
            num_data=x.shape[0],
        )
        gpf.set_trainable(self._model.inducing_variable, self.inducing_trainable)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_inducing: Optional[np.ndarray] = None,
        batch_size: int = 200,
        max_iter: int = 10000,
        learning_rate: float = 0.005,
        n_prints: int = 5,
        x_val: Optional[np.ndarray] = None,
        n_evals: int = 5,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        assert len(x.shape) == 2, "Input x with shape {} is not 2D.".format(x.shape)
        assert len(y.shape) == 2, "Input y with shape {} is not 2D.".format(y.shape)

        # Preprocess input data
        self._nc.compute_normalization_values(x, y)
        x, y = self._nc.normalize_and_clip(x, y)

        # TF Dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=x.shape[0])
        dataset = dataset.repeat()

        # Build model
        self._build(x, x_inducing, random_seed=random_seed)

        # Train
        self._train(
            dataset=dataset,
            batch_size=batch_size,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_prints=n_prints,
            x_val=x_val,
            n_evals=n_evals,
        )

    def _train(
        self,
        dataset: tf.data.Dataset,
        batch_size: int = 200,
        max_iter: int = 10000,
        learning_rate: float = 0.005,
        n_prints: int = 5,
        x_val: Optional[np.ndarray] = None,
        n_evals: int = 5,
    ) -> None:
        assert self._model is not None, "Model is not initialized."

        self.history = []
        iter_elbo = []
        train_iter = iter(dataset.batch(batch_size))
        training_loss = self._model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step():
            optimizer.minimize(training_loss, self._model.trainable_variables)

        for i in range(max_iter):
            train_step()
            elbo = -training_loss().numpy()
            iter_elbo.append(elbo)

            if (i + 1) % (max_iter // n_prints) == 0 or i == 0:
                pprint(
                    "- Step {:>6}/{}:".format(i + 1, max_iter),
                    "{:.3f}".format(elbo),
                    level=1,
                )

            if i % (max_iter // n_evals) == 0 and x_val is not None:
                y_out_mean, y_out_std = self.predict(x_val)
                self.history.append((y_out_mean, y_out_std))

        self.iter_elbo = np.array(iter_elbo)
