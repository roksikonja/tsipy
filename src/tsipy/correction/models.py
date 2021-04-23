from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import scipy.interpolate
import scipy.optimize
from qpsolvers import solve_qp
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
        out_of_bounds: str = "clip",
    ) -> None:
        """Initializes degradation model MRModel."""
        self.convex = None

        self._model = IsotonicRegression(
            y_max=y_max,
            y_min=y_min,
            increasing=False,
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
        out_of_bounds: str = "clip",
        n_pts: int = 999,
        lam: float = 10.0,
        solver: str = "quadprog",
        convex: bool = False,
    ) -> None:
        """Initializes degradation model SmoothMRModel."""
        self.solver = solver
        self.convex = convex

        self.n_pts = n_pts
        self.lam = lam

        self._model = None
        self._mr_model = IsotonicRegression(
            y_max=y_max,
            y_min=y_min,
            increasing=False,
            out_of_bounds=out_of_bounds,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model is not initialized."
        return self._model(x)

    def fit(self, x_a: np.ndarray, ratio: np.ndarray) -> None:
        self._model = self._solve_smooth_mr(x=x_a, y=ratio)

    def _solve_smooth_mr(self, x: np.ndarray, y: np.ndarray) -> Any:
        """Builds and solves smooth monotonic regression problem."""

        # To resolve numerical issues, we fit monotonic regression first
        self._mr_model.fit(x, y)
        x = np.linspace(0, np.max(x), self.n_pts)
        y = self._mr_model.predict(x)

        # Smooth monotonic regression
        # Linear cost
        q = -2 * y

        # Quadratic cost
        P = np.zeros((self.n_pts, self.n_pts))
        np.fill_diagonal(P, 1 + 2 * self.lam)
        P[0, 0] = 1 + self.lam
        P[-1, -1] = 1 + self.lam

        ids = np.arange(self.n_pts - 1)
        P[ids, ids + 1] = -self.lam
        P[ids + 1, ids] = -self.lam

        # Scale due to 1/2 in QP definition
        P = 2.0 * P

        # Inequality constraints
        # Monotonicity theta[i] >= theta[i + 1]
        G = np.zeros((self.n_pts - 1, self.n_pts))
        np.fill_diagonal(G, -1.0)
        ids = np.arange(self.n_pts - 1)
        G[ids, ids + 1] = 1.0
        h = np.zeros((self.n_pts - 1))

        # Convexity
        if self.convex:
            C = np.zeros((self.n_pts - 2, self.n_pts))
            np.fill_diagonal(C, -1.0)
            C[ids, ids + 1] = 2.0
            C[ids, ids + 2] = -1.0
            h_C = np.zeros((self.n_pts - 2))

            G = np.vstack((G, C))
            h = np.hstack((h, h_C))

        # Equality constraints
        # theta[0] = 1
        A = np.zeros((self.n_pts,))
        A[0] = 1.0
        b = np.array([1.0])

        # Upper and lower bounds of theta
        # 0.0 <= theta[i] <= 1.0
        lb = np.zeros((self.n_pts,))
        ub = np.ones((self.n_pts,))

        theta: np.ndarray = solve_qp(
            P=P,
            q=q,
            G=G,
            h=h,
            A=A,
            b=b,
            lb=lb,
            ub=ub,
            solver=self.solver,
            sym_proj=False,
            verbose=True,
        )

        model = scipy.interpolate.interp1d(x, theta, fill_value="extrapolate")
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
