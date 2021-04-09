from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import scipy.interpolate
import scipy.optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

__all__ = [
    "load_model",
    "DegradationModel",
    "ExpModel",
    "ExpLinModel",
    "MRModel",
    "SmoothMRModel",
]


class DegradationModel(ABC):
    """Abstract class for degradation model, that learns the degradation function.

    It must implement two methods :func:`˙__call__` for inference and :func:`fit` for
    training.

    For each degradation model, it must hold

    .. math::
        f(0, \\theta) = 1.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Inference method."""
        raise NotImplementedError

    def initial_fit(self, x_a: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> None:
        """Obtains an initial approximation for the model."""

    @abstractmethod
    def fit(self, x_a: np.ndarray, ratio: np.ndarray) -> None:
        """Learns a mapping from exposure to signal ratio.."""
        raise NotImplementedError


class ExpFamilyMixin:
    @staticmethod
    def _initial_fit(x_a: np.ndarray, ratio: np.ndarray) -> Tuple[float, float]:
        """Obtains an initial approximation.

        It learns a simple mapping

        .. math:: y = g(x, a, b) = e^{a \\cdot x + b},

        where .. math::`a < 0`.
        """
        epsilon = 1e-5
        gamma = np.min(ratio)

        x = x_a.reshape(-1, 1)
        y = np.log(ratio - gamma + epsilon)

        regression = LinearRegression(fit_intercept=True)
        regression.fit(x, y)

        lambda_ = -regression.coef_[0]
        e_0 = regression.intercept_ / lambda_

        return lambda_, e_0

    @staticmethod
    def _exp(x: np.ndarray, theta_1: float, theta_2: float) -> np.ndarray:
        """Implements function for :class:ExpModel.

        .. math::
            f(x, \\theta) = 1 - e^{\\theta_1 \\cdot \\theta_2}.
        """
        y = np.exp(-theta_1 * (x - theta_2)) + (1 - np.exp(theta_1 * theta_2))
        return y

    def _exp_lin(
        self, x: np.ndarray, theta_1: float, theta_2: float, theta_3: float
    ) -> np.ndarray:
        """Implements function for :class:ExpLinModel.

        .. math::
            f(x, \\theta) = 1 - e^{\\theta_1 \\cdot \\theta_2} + \\theta_3 \\cdot x.
        """
        y = self._exp(x, theta_1, theta_2) + theta_3 * x
        return y


class ExpModel(DegradationModel, ExpFamilyMixin):
    """Degradation model with prediction function in exponential form.

    Exact equation is:

    .. math::
        f(x, \\theta) = 1 - e^{\\theta_1 \\cdot \\theta_2}.
    """

    def __init__(self) -> None:
        """Initializes degradation model ExpModel."""
        self.convex = None

        self.initial_params = np.zeros((2,))
        self.params = np.zeros((2,))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._exp(x, *self.params)

    def initial_fit(self, x_a: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> None:
        ratio = np.divide(y_a, y_b + 1e-9)
        theta_1, theta_2 = self._initial_fit(x_a=x_a, ratio=ratio)

        self.params = np.array([theta_1, theta_2], dtype=np.float)

    def fit(self, x_a: np.ndarray, ratio: np.ndarray) -> None:
        params, _ = scipy.optimize.curve_fit(
            self._exp, x_a, ratio, p0=self.initial_params, maxfev=10000
        )
        self.params = params


class ExpLinModel(DegradationModel, ExpFamilyMixin):
    """Degradation model with prediction function in exponential form.

    Exact equation is:

    .. math::
        f(x, \\theta) = 1 - e^{\\theta_1 \\cdot \\theta_2} + \\theta_3 \\cdot x.
    """

    def __init__(self) -> None:
        """Initializes degradation model ExpLinModel."""
        self.convex = None

        self.initial_params = np.zeros((3,))
        self.params = np.zeros((3,))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._exp_lin(x, *self.params)

    def initial_fit(self, x_a: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> None:
        ratio_m = np.divide(y_a, y_b + 1e-9)
        theta_1, theta_2 = self._initial_fit(x_a=x_a, ratio=ratio_m)

        self.initial_params = np.array([theta_1, theta_2, 0.0], dtype=np.float)

    def fit(self, x_a: np.ndarray, ratio: np.ndarray) -> None:
        params, _ = scipy.optimize.curve_fit(
            self._exp_lin,
            x_a,
            ratio,
            p0=self.initial_params,
            maxfev=10000,
        )
        self.params = params


class MRModel(DegradationModel):
    def __init__(
        self,
        y_max: float = 1.0,
        y_min: float = 0.0,
        increasing: bool = False,
        out_of_bounds: str = "clip",
    ) -> None:
        """Initializes degradation model MRModel."""
        self.convex = None

        self._model = IsotonicRegression(
            y_max=y_max,
            y_min=y_min,
            increasing=increasing,
            out_of_bounds=out_of_bounds,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def fit(self, x_a: np.ndarray, ratio: np.ndarray) -> None:
        self._model.fit(x_a, ratio)


class SmoothMRModel(DegradationModel):
    def __init__(
        self,
        y_max: float = 1.0,
        y_min: float = 0.0,
        increasing: bool = False,
        out_of_bounds: str = "clip",
        number_of_points: int = 999,
        lam: float = 1.0,
        convex: bool = False,
    ) -> None:
        """Initializes degradation model SmoothMRModel."""
        self.convex = convex

        self.increasing = increasing
        self.number_of_points = number_of_points
        self.lam = lam

        self._model = None
        self._mr_model = IsotonicRegression(
            y_max=y_max,
            y_min=y_min,
            increasing=increasing,
            out_of_bounds=out_of_bounds,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model is not initialized."
        return self._model(x)

    def fit(self, x_a: np.ndarray, ratio: np.ndarray) -> None:
        self._mr_model.fit(x_a, ratio)

        x = np.linspace(0, np.max(x_a), self.number_of_points)
        y = self._mr_model.predict(x)

        self._model = self._solve_smooth_mr(x, y)

    def _solve_smooth_mr(self, x: np.ndarray, y: np.ndarray) -> Any:
        """Builds and solves smooth monotonic regression problem."""

        import cvxpy as cp  # TODO: Resolve conflict between cvxpy and tf.

        mu = cp.Variable(self.number_of_points)
        objective = cp.Minimize(
            cp.sum_squares(mu - y) + self.lam * cp.sum_squares(mu[:-1] - mu[1:])
        )

        constraints = [mu <= 1, mu[0] == 1]
        if not self.increasing:
            constraints.append(mu[1:] <= mu[:-1])

        if self.convex:
            constraints.append(mu[:-2] + mu[2:] >= 2 * mu[1:-1])

        model = cp.Problem(objective, constraints)
        model.solve(solver=cp.ECOS_BB)

        model = scipy.interpolate.interp1d(x, mu.value, fill_value="extrapolate")
        return model


def load_model(model: str) -> DegradationModel:
    """Loads a degradation model given the model name.

    Currently, the following models are implemented
        - exponential ``exp``,
        - exponential with linear term ``explin``,
        - monotonic regression ``mr``, and
        - smooth monotonic regression ``smr``.

    Args:
        model: String abbreviation of the model name.
    """
    model = model.lower()
    if model == "exp":
        return ExpModel()
    elif model == "explin":
        return ExpLinModel()
    elif model == "mr":
        return MRModel()
    elif model == "smr":
        return SmoothMRModel()

    raise ValueError("Invalid degradation model type.")
