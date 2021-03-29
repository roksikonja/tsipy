from typing import Optional, Tuple, List, NoReturn, Union

import gpflow as gpf
import numpy as np
import tensorflow as tf
from gpflow.utilities.utilities import tabulate_module_summary, default_summary_fmt
from tsipy.fusion.core import NormalizationClippingMixin, FusionModel
from tsipy.utils import (
    normalize,
    denormalize,
    find_nearest_indices,
    pprint,
)


class SVGPModel(NormalizationClippingMixin, FusionModel):
    def __init__(
        self,
        kernel: gpf.kernels.Kernel,
        num_inducing_pts: int = 1000,
        inducing_trainable: bool = False,
        normalization: bool = True,
        clipping: bool = True,
    ):
        super(SVGPModel, self).__init__(normalization=normalization, clipping=clipping)
        self._model: Optional[gpf.models.GPModel] = None
        self._kernel = kernel
        self.num_inducing_pts = num_inducing_pts
        self.inducing_trainable = inducing_trainable

        self.iter_elbo: Optional[np.ndarray] = None
        self.x_inducing: Optional[np.ndarray] = None
        self.t_prior: Optional[np.ndarray] = None
        self.t_posterior: Optional[np.ndarray] = None

        self.history: List = []

    def __call__(
        self, x: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x: Union[np.ndarray, tf.Tensor] = normalize(x.copy(), self.x_mean, self.x_std)
        y_mean, y_var = self._model.predict_y(x)

        y_mean = y_mean.numpy()
        y_std = np.sqrt(y_var.numpy())

        y_mean = denormalize(y_mean, self.y_mean, self.y_std)
        y_std = denormalize(y_std, 0.0, self.y_std)  # standard deviation scaling
        return y_mean.ravel(), y_std.ravel()

    def build_model(
        self,
        x: np.ndarray,
        x_inducing: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ) -> NoReturn:
        self.set_random_seed(random_seed)

        # Inducing variables
        if x_inducing is None:
            x_uniform = np.linspace(
                np.min(x[:, 0]), np.max(x[:, 0]), self.num_inducing_pts + 1
            )
            x_uniform_indices = find_nearest_indices(x[:, 0], x_uniform)
            x_inducing = x[x_uniform_indices, :].copy()
            self.x_inducing = x_inducing
        else:
            self.x_inducing = x_inducing

        if self._model is None:
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
    ) -> NoReturn:

        x, y = self.normalize_and_clip(x, y)

        x = np.atleast_2d(x)
        y = y.reshape(-1, 1)

        # TF Dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=x.shape[0])
        dataset = dataset.repeat()

        self.build_model(x, x_inducing, random_seed=random_seed)

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

    def train(
        self,
        dataset: tf.data.Dataset,
        batch_size: int = 200,
        max_iter: int = 10000,
        learning_rate: float = 0.005,
        n_prints: int = 5,
        x_val: Optional[np.ndarray] = None,
        n_evals: int = 5,
    ) -> NoReturn:
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
                y_out_mean, y_out_std = self(x_val)
                self.history.append((y_out_mean, y_out_std))

        self.iter_elbo = np.array(iter_elbo)

    def __str__(self) -> str:
        return tabulate_module_summary(self._model, default_summary_fmt())

    @staticmethod
    def set_random_seed(random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
