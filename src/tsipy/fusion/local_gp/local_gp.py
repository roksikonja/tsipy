import copy
from typing import Tuple, NoReturn, Optional

import numpy as np
import tensorflow as tf

from .windows import Windows
from ..core import FusionModel, NormalizationClippingMixin
from ...utils import pprint


class LocalGPModel(NormalizationClippingMixin):
    def __init__(
        self,
        model: FusionModel,
        normalization: bool = True,
        clipping: bool = True,
    ):
        super(LocalGPModel, self).__init__(
            normalization=normalization, clipping=clipping
        )
        self._model = model
        self.windows: Optional[Windows] = None

    def __call__(
        self, x: np.ndarray, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize
        pred_windows_ids = self.windows.create_prediction_windows_ids(x)

        y_mean, y_std = [], []
        for window_id, window in enumerate(self.windows):
            start_id, end_id = pred_windows_ids[window_id]

            x_window = x[start_id : end_id + 1, :]
            y_window_mean, y_window_std = self.predict_window(
                x=x_window, window_id=window_id, verbose=verbose
            )
            y_mean.append(y_window_mean)
            y_std.append(y_window_std)

            if verbose:
                pprint("- Indices:", start_id, end_id, level=1)

        y_mean = np.hstack(y_mean)
        y_std = np.hstack(y_std)
        return y_mean, y_std

    def predict_window(
        self, x: np.ndarray, window_id: int, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert (
            0 <= window_id < len(self.windows)
        ), "Window index {} is out of bounds [0, {}].".format(
            window_id, len(self.windows) - 1
        )

        window = self.windows[window_id]
        model = window.model

        # Window prediction
        y_window_mean, y_window_std = model(x)

        if verbose:
            print(str(window) + "\n")
            pprint("- x_window:", x.shape, level=1)
            pprint("- y_mean:", y_window_mean.shape, level=1)
            pprint("- y_std:", y_window_std.shape, level=1)

        return y_window_mean, y_window_std

    def build_models(self, random_seed : Optional[int] = None) -> NoReturn:
        x, y = self.windows.gather_data()
        self._compute_normalization_values(x[:, 0], y)

        self._model.normalization = self.normalization
        self._model.clipping = self.clipping

        for window in self.windows:
            window.model = copy.deepcopy(self._model)
            window.model.build_model(window.x, window.x_inducing, random_seed=random_seed)

            # Update normalize values of the model
            window.model.x_mean = self.x_mean
            window.model.x_std = self.x_std
            window.model.y_mean = self.y_mean
            window.model.y_std = self.y_std

    def fit(
        self,
        windows: Windows,
        n_prints: int = 5,
        random_seed: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ) -> NoReturn:
        self.windows = windows
        self.build_models(random_seed=random_seed)

        for window in self.windows:
            if verbose:
                print(str(window) + "\n")

            model = window.model
            x_window, y_window = self.normalize_and_clip(window.x, window.y)
            x_window = np.atleast_2d(x_window)
            y_window = y_window.reshape(-1, 1)

            # TF Dataset
            dataset = tf.data.Dataset.from_tensor_slices((x_window, y_window))
            dataset = dataset.shuffle(buffer_size=window.x.shape[0])
            dataset = dataset.repeat()

            # Train
            model.train(
                dataset=dataset, n_prints=n_prints, x_val=window.x_val, **kwargs
            )
