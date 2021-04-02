from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from ..utils import nonclipped_indices, normalize, denormalize


class FusionModel(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def set_random_seed(random_seed: Optional[int] = None) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)


class NormalizeAndClip:
    """Normalizes and clips data in the first dimension.

    Normalization enables easier and more stable learning of the GP.

    Clipping discards outliers, which could disturb the learning process.
    """

    def __init__(self, normalization: bool = True, clipping: bool = True) -> None:
        self.normalization = normalization
        self.clipping = clipping

        self._x_shift: Optional[float] = None
        self._x_scale: Optional[float] = None
        self._y_shift: Optional[float] = None
        self._y_scale: Optional[float] = None

    @property
    def x_shift(self):
        assert self._x_shift is not None, "Normalization values not computed."
        return self._x_shift

    @property
    def x_scale(self):
        assert self._x_scale is not None, "Normalization values not computed."
        return self._x_scale

    @property
    def y_shift(self):
        assert self._y_shift is not None, "Normalization values not computed."
        return self._y_shift

    @property
    def y_scale(self):
        assert self._y_scale is not None, "Normalization values not computed."
        return self._y_scale

    @property
    def initialized(self):
        return (
            self._x_shift is not None
            or self._x_scale is not None
            or self._y_shift is not None
            or self._y_scale is not None
        )

    def normalize_x(
        self,
        x: np.ndarray,
        x_shift: Optional[float] = None,
        x_scale: Optional[float] = None,
    ):
        self._assert_2d(x)

        x = np.copy(x)  # Prevent inplace modification
        x_shift = self.x_shift if x_shift is None else x_shift
        x_scale = self.x_scale if x_scale is None else x_scale

        x[:, 0] = normalize(x[:, 0], x_shift, x_scale)
        return x

    def normalize_y(
        self,
        y: np.ndarray,
        y_shift: Optional[float] = None,
        y_scale: Optional[float] = None,
    ):
        self._assert_2d(y)

        y = np.copy(y)  # Prevent inplace modification
        y_shift = self.y_shift if y_shift is None else y_shift
        y_scale = self.y_scale if y_scale is None else y_scale

        y = normalize(y, y_shift, y_scale)
        return y

    def denormalize_x(
        self,
        x: np.ndarray,
        x_shift: Optional[float] = None,
        x_scale: Optional[float] = None,
    ):
        self._assert_2d(x)

        x = np.copy(x)  # Prevent inplace modification
        x_shift = self.x_shift if x_shift is None else x_shift
        x_scale = self.x_scale if x_scale is None else x_scale

        x[:, 0] = denormalize(x[:, 0], x_shift, x_scale)
        return x

    def denormalize_y(
        self,
        y: np.ndarray,
        y_shift: Optional[float] = None,
        y_scale: Optional[float] = None,
    ):
        self._assert_2d(y)

        y = np.copy(y)  # Prevent inplace modification
        y_shift = self.y_shift if y_shift is None else y_shift
        y_scale = self.y_scale if y_scale is None else y_scale

        y = denormalize(y, y_shift, y_scale)
        return y

    def normalize_and_clip(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._assert_2d(x)
        self._assert_2d(y)

        x = self.normalize_x(x)
        y = self.normalize_y(y)
        x, y = self.clip_by_y_values(x, y)
        return x, y

    def clip_by_y_values(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._assert_2d(x)
        self._assert_2d(y)

        if self.clipping:
            nonclipped_ids = nonclipped_indices(y[:, 0])
            x, y = x[nonclipped_ids, :], y[nonclipped_ids, :]

        return x, y

    def compute_normalization_values(self, x: np.ndarray, y: np.ndarray) -> None:
        self._assert_2d(x)
        self._assert_2d(y)
        assert (
            y.shape[1] == 1
        ), "Input y with shape {} must have exactly 1 column.".format(y.shape)

        if self.initialized:
            raise ValueError("Attempted to re-compute normalization values.")

        if self.normalization:
            self._x_shift = np.mean(x[:, 0])
            self._x_scale = np.std(x[:, 0])
            self._y_shift = np.mean(y)
            self._y_scale = np.std(y)
        else:
            self._x_shift = 0.0
            self._x_scale = 1.0
            self._y_shift = 0.0
            self._y_scale = 1.0

    def reset(self):
        self._x_shift = None
        self._x_scale = None
        self._y_shift = None
        self._y_scale = None

    @staticmethod
    def _assert_2d(x: np.ndarray) -> None:
        assert len(x.shape) == 2, "Input x with shape {} is not 2D.".format(x.shape)
