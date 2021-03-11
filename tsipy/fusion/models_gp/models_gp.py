from abc import ABC, abstractmethod

import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow.utilities.utilities import tabulate_module_summary, default_summary_fmt

from ...utils import (
    normalize,
    denormalize,
    find_nearest_indices,
    clipping_indices,
    pprint,
)


class BaseOutputModel(ABC):
    @abstractmethod
    def __call__(self, t):
        pass

    @abstractmethod
    def fit(self, x, y, **kwargs):
        pass


class NormalizationClippingMixin:
    def __init__(self, normalization, clipping):
        self.normalization = normalization
        self.clipping = clipping

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def _compute_normalization_values(self, x, y):
        if self.normalization:
            self.x_mean = np.mean(x)
            self.x_std = np.std(x)
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
        else:
            self.x_mean = 0.0
            self.x_std = 1.0
            self.y_mean = np.mean(y)
            self.y_std = 1.0


class SVGPModel(NormalizationClippingMixin, BaseOutputModel):
    def __init__(
        self,
        kernel,
        num_inducing_pts=1000,
        inducing_trainable=False,
        normalization=True,
        clipping=True,
    ):
        super(SVGPModel, self).__init__(normalization=normalization, clipping=clipping)
        self._model = None
        self._kernel = kernel
        self.num_inducing_pts = num_inducing_pts
        self.inducing_trainable = inducing_trainable

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.iter_elbo = None
        self.x_inducing = None
        self.t_prior = None
        self.t_posterior = None

        self.history = []

    def __call__(self, x):
        x = normalize(x.copy(), self.x_mean, self.x_std)
        y_mean, y_var = self._model.predict_y(x)

        y_mean = y_mean.numpy()
        y_std = np.sqrt(y_var.numpy())

        y_mean = denormalize(y_mean, self.y_mean, self.y_std)
        y_std = denormalize(y_std, 0.0, self.y_std)  # standard deviation scaling
        return y_mean.ravel(), y_std.ravel()

    def _build_model(self, x, x_inducing):
        # Inducing variables
        if isinstance(x_inducing, type(None)):
            x_uniform = np.linspace(
                np.min(x[:, 0]), np.max(x[:, 0]), self.num_inducing_pts + 1
            )
            x_uniform_indices = find_nearest_indices(x[:, 0], x_uniform)
            x_inducing = x[x_uniform_indices, :].copy()
            self.x_inducing = x_inducing
        else:
            self.x_inducing = x_inducing

        if isinstance(self._model, type(None)):
            # Build SVGP model
            self._model = gpf.models.SVGP(
                self._kernel,
                gpf.likelihoods.Gaussian(),
                self.x_inducing,
                num_data=x.shape[0],
            )
            gpf.set_trainable(self._model.inducing_variable, self.inducing_trainable)

    def fit(
        self,
        x,
        y,
        x_inducing=None,
        batch_size=200,
        max_iter=10000,
        learning_rate=0.005,
        n_prints=100,
        x_val=None,
        n_evals=5,
        verbose=False,
    ):
        self._compute_normalization_values(x, y)
        x = normalize(x.copy(), self.x_mean, self.x_std)
        y = normalize(y.copy(), self.y_mean, self.y_std)

        if self.clipping:
            clip_indices = clipping_indices(y)
            x, y = x[clip_indices, :], y[clip_indices]

        x = np.atleast_2d(x)
        y = y.reshape(-1, 1)

        # TF Dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=x.shape[0])

        self._build_model(x, x_inducing)

        # Train
        self.train(
            dataset=dataset,
            batch_size=batch_size,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_prints=n_prints,
            x_val=x_val,
            n_evals=n_evals,
        )

        if verbose:
            self.print()

    def train(
        self,
        dataset,
        batch_size=200,
        max_iter=10000,
        learning_rate=0.005,
        n_prints=100,
        x_val=None,
        n_evals=5,
    ):
        iter_elbo = []
        train_iter = iter(dataset.batch(batch_size))
        training_loss = self._model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step():
            optimizer.minimize(training_loss, self._model.trainable_variables)

        i = None
        elbo = None
        for i in range(max_iter):
            train_step()
            elbo = -training_loss().numpy()
            iter_elbo.append(elbo)

            if i % (max_iter // n_prints) == 0:
                pprint(
                    "    - Step {:>6}/{}:".format(i, max_iter), "{:.3f}".format(elbo)
                )

            if i % (max_iter // n_evals) == 0 and x_val is not None:
                y_out_mean, y_out_std = self(x_val)
                self.history.append((y_out_mean, y_out_std))

        pprint("    - Step {:>6}/{}:".format(i, max_iter), "{:.3f}".format(elbo))

        self.iter_elbo = np.array(iter_elbo)

    def __str__(self):
        return tabulate_module_summary(self._model, default_summary_fmt())

    def print(self):
        print(self)
