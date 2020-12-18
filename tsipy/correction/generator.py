import numpy as np
import pandas as pd
from scipy.stats import norm

from .exposure import ExposureMethod, compute_exposure
from .models import ExpModel


class GeneratorConstants:
    SIGNAL_LENGTH = 100000
    DOWNSAMPLING_A = 0.9
    DOWNSAMPLING_B = 0.2
    STD_NOISE_A = 0.025
    STD_NOISE_B = 0.015


class SignalGenerator(object):
    def __init__(
        self,
        length=GeneratorConstants.SIGNAL_LENGTH,
        downsampling_a=GeneratorConstants.DOWNSAMPLING_A,
        downsampling_b=GeneratorConstants.DOWNSAMPLING_B,
        std_noise_a=GeneratorConstants.STD_NOISE_A,
        std_noise_b=GeneratorConstants.STD_NOISE_B,
        exposure_method=ExposureMethod.NUM_MEASUREMENTS,
        random_seed=0,
    ):
        np.random.seed(random_seed)

        self.downsampling_a = downsampling_a
        self.downsampling_b = downsampling_b
        self.std_noise_a = std_noise_a
        self.std_noise_b = std_noise_b

        self.t = np.linspace(0, 1, length)
        self.s = self.generate_signal()

        self.a, self.b = None, None
        self.t_a_indices = None  # Measurement time indices of each sensor
        self.t_b_indices = None  # Measurement time indices of each sensor
        self.e_a, self.e_b = None, None  # Exposure of each sensor

        self.exposure_method = exposure_method
        self.degradation_model = ExpModel()
        self.degradation_model.params = self.compute_degradation_params()

        self.generate_measurements()

    @property
    def data(self):
        data = pd.DataFrame(
            {
                "t": self.t,
                "a": self.a,
                "b": self.b,
            }
        )
        return data

    def generate_signal(self):
        x = 10 + self._brownian(self.t.shape[0], dt=self.t[1] - self.t[0], std_scale=5)
        return x

    def generate_measurements(self):
        srange = self.s.max() - self.s.min()
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
        self.e_a = self.e_a / max_e
        self.e_b = self.e_b / max_e

        d_a_nn = self.degradation_model(self.e_a[self.t_a_indices])
        d_b_nn = self.degradation_model(self.e_b[self.t_b_indices])

        noise_a_nn = self.generate_noise(
            self.a[self.t_a_indices].shape, std=srange * self.std_noise_a
        )
        noise_b_nn = self.generate_noise(
            self.b[self.t_b_indices].shape, std=srange * self.std_noise_b
        )

        # Degrade signals
        self.a[self.t_a_indices] *= d_a_nn
        self.b[self.t_b_indices] *= d_b_nn

        # Add noise
        self.a[self.t_a_indices] += noise_a_nn
        self.b[self.t_b_indices] += noise_b_nn

    @staticmethod
    def _brownian(n, dt, std_scale):
        # x = np.random.normal(loc=0.0, scale=std_scale * np.sqrt(dt), size=(n, ))
        x = norm.rvs(size=(n,), scale=std_scale * np.sqrt(dt), loc=0.0)
        return np.cumsum(x, axis=-1)

    @staticmethod
    def measurement_indices(t, rate):
        nn_indices = np.random.rand(t.shape[0]) <= rate
        return nn_indices

    @staticmethod
    def compute_degradation_params():
        params = np.random.rand(2)
        params[1] = -np.abs(params[1])
        return params

    @staticmethod
    def generate_noise(shape, std=1.0):
        noise = np.random.normal(0, std, shape)
        return noise
