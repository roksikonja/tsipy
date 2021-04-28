import string
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from .exposure import compute_exposure
from .models import ExpModel

__all__ = ["SignalGenerator"]


class SignalGenerator:
    """Class for generating sample signals."""

    def __init__(
        self,
        length: int = 100000,
        y_center: float = 10.0,
        add_degradation: bool = True,
        add_noise: bool = True,
        downsampling_rates: Tuple = (0.9, 0.2),
        noise_stds: Tuple = (0.025, 0.015),
        exposure_method: str = "num_measurements",
        random_seed: int = 0,
    ):
        np.random.seed(random_seed)
        self.length = length

        self.y_center = y_center

        self.add_degradation = add_degradation
        self.add_noise = add_noise

        self.downsampling_rates = downsampling_rates
        self.noise_stds = noise_stds

        self.x: np.ndarray = np.linspace(0.0, 1.0, self.length)
        self.y: np.ndarray = self.generate_signal()

        self.signal_names: List[str] = list(string.ascii_lowercase[:26])
        self.signals: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = dict()

        self.exposure_method = exposure_method
        self.degradation_model = ExpModel()
        self.degradation_model.params = self.generate_degradation_parameters()

    @property
    def data(self) -> pd.DataFrame:
        """Two."""
        a = self._get_signal("a")
        b = self._get_signal("b")
        data = pd.DataFrame(
            {
                "t": self.x,
                "a": a[0],
                "b": b[0],
            }
        )
        return data

    def __getitem__(self, signal_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Gets non-nan x and y of `key` signal."""
        y, x_indices, _ = self._get_signal(signal_name)
        return self.x[x_indices], y[x_indices]

    def get_exposure_nn(self, signal_name: str) -> np.ndarray:
        _, x_indices, e = self._get_signal(signal_name)
        return e[x_indices]

    def get_indices_nn(self, signal_name: str) -> np.ndarray:
        return self._get_signal(signal_name)[1]

    def get_signal_nn(
        self, signal_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y, x_indices, e = self._get_signal(signal_name)

        x_nn = self.x[x_indices]
        y_nn = y[x_indices]
        e_nn = e[x_indices]
        return x_nn, y_nn, e_nn

    def _get_signal(
        self, signal_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if signal_name not in self.signals:
            self.generate_measurement(signal_name)

        return self.signals[signal_name]

    def _get_signal_parameters(self, signal_name: str) -> Tuple[float, float]:
        signal_id = self.signal_names.index(signal_name)

        if signal_id < len(self.downsampling_rates):
            downsampling_rate = self.downsampling_rates[signal_id]
        else:
            downsampling_rate = 0.5

        if signal_id < len(self.noise_stds):
            noise_std = self.noise_stds[signal_id]
        else:
            noise_std = 0.02

        return downsampling_rate, noise_std

    def generate_measurement(
        self, signal_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        downsampling_rate, noise_std = self._get_signal_parameters(signal_name)

        y_range = np.max(self.y) - np.min(self.y)
        y_mean = float(np.mean(self.y))

        # Copy ground truth signal
        y = self.y.copy()

        # Get measurement time indices
        x_indices = self.generate_measurement_indices(self.x, downsampling_rate)
        y[~x_indices] = np.nan

        # Degrade signals
        e = np.zeros_like(y)
        if self.add_degradation:
            e = compute_exposure(y, method=self.exposure_method, x_mean=y_mean)
            e /= self.length  # Normalize exposure to [0, 1]
            y[x_indices] *= self.degradation_model(e[x_indices])

        # Add noise
        if self.add_noise:
            y[x_indices] += self.generate_noise(y[x_indices], std=y_range * noise_std)

        self.signals[signal_name] = (y, x_indices, e)
        return self.signals[signal_name]

    def generate_signal(self) -> np.ndarray:
        y = self.y_center + self.generate_brownian(
            self.x.shape[0], dt=self.x[1] - self.x[0], std_scale=5
        )
        return y

    @staticmethod
    def generate_brownian(n: int, dt: float, std_scale: float) -> np.ndarray:
        y = scipy.stats.norm.rvs(size=(n,), scale=std_scale * np.sqrt(dt), loc=0.0)
        return np.cumsum(y)

    @staticmethod
    def generate_degradation_parameters() -> np.ndarray:
        params = np.random.rand(2)
        params[1] = -np.abs(params[1])
        return params

    @staticmethod
    def generate_measurement_indices(t: np.ndarray, rate: float) -> np.ndarray:
        nn_indices = np.random.rand(t.shape[0]) <= rate
        return nn_indices

    @staticmethod
    def generate_noise(y: np.ndarray, std: float = 1.0) -> np.ndarray:
        shape = y.shape
        noise = np.random.normal(0, std, shape)
        return noise
