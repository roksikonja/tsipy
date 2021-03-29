from abc import ABC, abstractmethod
from typing import Optional, NoReturn, Tuple

import numpy as np

from ..utils import clipping_indices, normalize


class FusionModel(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> NoReturn:
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs) -> NoReturn:
        pass


class NormalizationClippingMixin:
    def __init__(self, normalization: bool, clipping: bool):
        self.normalization = normalization
        self.clipping = clipping

        self.x_mean: Optional[float] = None
        self.x_std: Optional[float] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

    def normalize_and_clip(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._compute_normalization_values(x, y)
        x = normalize(x.copy(), self.x_mean, self.x_std)
        y = normalize(y.copy(), self.y_mean, self.y_std)
        x, y = self._clip_y_values(x, y)

        return x, y

    def _compute_normalization_values(self, x: np.ndarray, y: np.ndarray) -> NoReturn:
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

    def _clip_y_values(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.clipping:
            clip_indices = clipping_indices(y)
            x, y = x[clip_indices, :], y[clip_indices]

        return x, y
