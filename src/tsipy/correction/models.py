from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import scipy.interpolate
import scipy.optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


class DegradationModel(ABC):
    @abstractmethod
    def __call__(self, e: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def initial_fit(self, a_m: np.ndarray, b_m: np.ndarray, e_a_m: np.ndarray) -> None:
        pass

    @abstractmethod
    def fit(self, ratio_m: np.ndarray, e_a_m: np.ndarray) -> None:
        pass


class ExpFamilyMixin:
    @staticmethod
    def _initial_fit(ratio: np.ndarray, e_a_m: np.ndarray) -> Tuple[float, float]:
        epsilon = 1e-5
        gamma = np.min(ratio)

        y = np.log(ratio - gamma + epsilon)
        x = e_a_m.reshape(-1, 1)

        regression = LinearRegression(fit_intercept=True)
        regression.fit(x, y)

        lambda_ = -regression.coef_[0]
        e_0 = regression.intercept_ / lambda_

        return lambda_, e_0

    @staticmethod
    def _exp(x: np.ndarray, lambda_: float, e_0: float) -> np.ndarray:
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        lambda_ : float
            Rate of exponential decay.
        e_0 : float
            Offset of exponential decay

        Returns
        -------
        y : array_like
            Returns 1 - exp(lambda_ * e_0) + exp(-lambda_ * (x - e_0)). It holds y(0) = 1.

        """
        y = np.exp(-lambda_ * (x - e_0)) + (1 - np.exp(lambda_ * e_0))
        return y

    def _exp_lin(
        self, x: np.ndarray, lambda_: float, e_0: float, linear: float
    ) -> np.ndarray:
        """

        Parameters
        ----------
        x : array_like
            Input x coordinate values.
        lambda_ : float
            Rate of exponential decay.
        e_0 : float
            Offset of exponential decay
        linear : float
            Coefficient of linear decay.

        Returns
        -------
        y : array_like
            Returns 1 - exp(lambda_ * e_0) + exp(-lambda_ * (x - e_0)) + linear * x. It holds y(0) = 1.

        """
        y = self._exp(x, lambda_, e_0) + linear * x
        return y


class ExpModel(DegradationModel, ExpFamilyMixin):
    def __init__(self):
        self.name = "Exp"
        self.convex = None

        self.initial_params = np.zeros((2,))
        self.params = np.zeros((2,))

    def __call__(self, e: np.ndarray) -> np.ndarray:
        return self._exp(e, *self.params)

    def initial_fit(self, a_m: np.ndarray, b_m: np.ndarray, e_a_m: np.ndarray) -> None:
        ratio_m = np.divide(a_m, b_m)
        lambda_initial, e_0_initial = self._initial_fit(ratio_m, e_a_m)

        self.params = np.array([lambda_initial, e_0_initial], dtype=np.float)

    def fit(self, ratio_m: np.ndarray, e_a_m: np.ndarray) -> None:
        params, _ = scipy.optimize.curve_fit(
            self._exp, e_a_m, ratio_m, p0=self.initial_params, maxfev=10000
        )
        self.params = params

    def __repr__(self) -> str:
        return self.name


class ExpLinModel(DegradationModel, ExpFamilyMixin):
    def __init__(self):
        self.name = "ExpLin"
        self.convex = None

        self.initial_params = np.zeros((3,))
        self.params = np.zeros((3,))

    def __call__(self, e: np.ndarray) -> np.ndarray:
        return self._exp_lin(e, *self.params)

    def initial_fit(self, a_m: np.ndarray, b_m: np.ndarray, e_a_m: np.ndarray) -> None:
        ratio_m = np.divide(a_m, b_m)
        lambda_initial, e_0_initial = self._initial_fit(ratio_m, e_a_m)

        self.initial_params = np.array(
            [lambda_initial, e_0_initial, 0.0], dtype=np.float
        )

    def fit(self, ratio_m: np.ndarray, e_a_m: np.ndarray) -> None:
        params, _ = scipy.optimize.curve_fit(
            self._exp_lin,
            e_a_m,
            ratio_m,
            p0=self.initial_params,
            maxfev=10000,
        )
        self.params = params

    def __repr__(self) -> str:
        return self.name


class MRModel(DegradationModel):
    def __init__(
        self,
        y_max: float = 1.0,
        y_min: float = 0.0,
        increasing: bool = False,
        out_of_bounds: str = "clip",
    ):
        self.name = "MR"
        self.convex = None

        self._model = IsotonicRegression(
            y_max=y_max,
            y_min=y_min,
            increasing=increasing,
            out_of_bounds=out_of_bounds,
        )

    def initial_fit(self, a_m: np.ndarray, b_m: np.ndarray, e_a_m: np.ndarray) -> None:
        pass

    def __call__(self, e: np.ndarray) -> np.ndarray:
        return self._model.predict(e)

    def fit(self, ratio_m: np.ndarray, e_a_m: np.ndarray) -> None:
        self._model.fit(e_a_m, ratio_m)

    def __repr__(self) -> str:
        return self.name


class SmoothMRModel(DegradationModel):
    def __init__(
        self,
        y_max: float = 1.0,
        y_min: float = 0.0,
        increasing: bool = False,
        out_of_bounds: str = "clip",
        number_of_points: int = 999,
        lam: float = 1.0,
    ):
        self.name = "SmoothMR"
        self.convex = True

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

    def initial_fit(self, a_m: np.ndarray, b_m: np.ndarray, e_a_m: np.ndarray) -> None:
        pass

    def __call__(self, e: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model is not initialized."
        return self._model(e)

    def fit(self, ratio_m: np.ndarray, e_a_m: np.ndarray) -> None:
        self._mr_model.fit(e_a_m, ratio_m)

        max_exposure = e_a_m[-1]
        x = np.linspace(0, max_exposure, self.number_of_points)
        y = self._mr_model.predict(x)

        self._model = self._solve_smooth_mr(x, y)

    def _solve_smooth_mr(self, x: np.ndarray, y: np.ndarray) -> Any:
        import cvxpy as cp

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

    def __repr__(self) -> str:
        return self.name


def load_model(degradation_model: str) -> DegradationModel:
    if degradation_model == "exp":
        return ExpModel()
    elif degradation_model == "explin":
        return ExpLinModel()
    elif degradation_model == "mr":
        return MRModel()
    elif degradation_model == "smr":
        return SmoothMRModel()
    else:
        raise ValueError("Invalid degradation model type.")
