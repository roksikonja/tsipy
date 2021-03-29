from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from .exposure import compute_exposure
from .models import ExpModel


class SignalGenerator(object):
    def __init__(
        self,
        length: int = 100000,
        add_degradation: bool = True,
        add_noise: bool = True,
        downsampling_a: float = 0.9,
        downsampling_b: float = 0.2,
        std_noise_a: float = 0.025,
        std_noise_b: float = 0.015,
        exposure_method: str = "num_measurements",
        random_seed: int = 0,
    ):
        np.random.seed(random_seed)

        self.add_degradation = add_degradation
        self.add_noise = add_noise

        self.downsampling_a = downsampling_a
        self.downsampling_b = downsampling_b
        self.std_noise_a = std_noise_a
        self.std_noise_b = std_noise_b

        self.t: np.ndarray = np.linspace(0, 1, length)
        self.s: np.ndarray = self.generate_signal()

        self.a: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.t_a_indices: Optional[
            np.ndarray
        ] = None  # Measurement time indices of each sensor
        self.t_b_indices: Optional[
            np.ndarray
        ] = None  # Measurement time indices of each sensor

        self.e_a: Optional[np.ndarray] = None
        self.e_b: Optional[np.ndarray] = None

        self.exposure_method = exposure_method
        self.degradation_model = ExpModel()
        self.degradation_model.params = self.compute_degradation_params()

        self.generate_measurements()

    @property
    def data(self) -> pd.DataFrame:
        data = pd.DataFrame(
            {
                "t": self.t,
                "a": self.a,
                "b": self.b,
            }
        )
        return data

    def generate_signal(self) -> np.ndarray:
        x = 10 + self._brownian(self.t.shape[0], dt=self.t[1] - self.t[0], std_scale=5)
        return x

    def generate_measurements(self):
        s_range = np.max(self.s) - np.min(self.s)
        s_mean = float(np.mean(self.s))

        # Get measurement time indices
        self.t_a_indices = self.measurement_indices(self.t, self.downsampling_a)
        self.t_b_indices = self.measurement_indices(self.t, self.downsampling_b)

        self.a = self.s.copy()
        self.a[~self.t_a_indices] = np.nan
        self.b = self.s.copy()
        self.b[~self.t_b_indices] = np.nan

        self.e_a = compute_exposure(self.a, method=self.exposure_method, x_mean=s_mean)
        self.e_b = compute_exposure(self.b, method=self.exposure_method, x_mean=s_mean)

        max_e = max(np.max(self.e_a), np.max(self.e_b))
        self.e_a /= max_e
        self.e_b /= max_e

        d_a_nn = self.degradation_model(self.e_a[self.t_a_indices])
        d_b_nn = self.degradation_model(self.e_b[self.t_b_indices])

        noise_a_nn = self.generate_noise(
            self.a[self.t_a_indices].shape, std=s_range * self.std_noise_a
        )
        noise_b_nn = self.generate_noise(
            self.b[self.t_b_indices].shape, std=s_range * self.std_noise_b
        )

        # Degrade signals
        if self.add_degradation:
            self.a[self.t_a_indices] *= d_a_nn
            self.b[self.t_b_indices] *= d_b_nn

        # Add noise
        if self.add_noise:
            self.a[self.t_a_indices] += noise_a_nn
            self.b[self.t_b_indices] += noise_b_nn

    @staticmethod
    def _brownian(n: int, dt: float, std_scale: float) -> np.ndarray:
        x = scipy.stats.norm.rvs(size=(n,), scale=std_scale * np.sqrt(dt), loc=0.0)
        return np.cumsum(x, axis=-1)

    @staticmethod
    def measurement_indices(t: np.ndarray, rate: float) -> np.ndarray:
        nn_indices = np.random.rand(t.shape[0]) <= rate
        return nn_indices

    @staticmethod
    def compute_degradation_params() -> np.ndarray:
        params = np.random.rand(2)
        params[1] = -np.abs(params[1])
        return params

    @staticmethod
    def generate_noise(shape: Tuple[int, ...], std: float = 1.0) -> np.ndarray:
        noise = np.random.normal(0, std, shape)
        return noise
