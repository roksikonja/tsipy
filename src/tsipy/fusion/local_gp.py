import copy
from typing import Optional, Tuple

import numpy as np

from tsipy.utils import pprint
from tsipy.fusion.core import FusionModel, NormalizeAndClip
from .windows import Windows, create_windows


class LocalGPModel(FusionModel):
    def __init__(
        self,
        model: FusionModel,
        pred_window_width: float,
        fit_window_width: float,
        normalization: bool = True,
        clipping: bool = True,
    ) -> None:
        # Helper object for data normalization and clipping
        self._nc = NormalizeAndClip(normalization=normalization, clipping=clipping)

        # Local windows attributes and parameters
        self.pred_window_width = pred_window_width
        self.fit_window_width = fit_window_width
        self._windows: Optional[Windows] = None

        # Local GP model that is trained within each window
        self._model = model

    @property
    def windows(self) -> Windows:
        assert self._windows is not None, "Windows are not initialized."
        return self._windows

    def __call__(
        self, x: np.ndarray, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Equivalent to :member:`predict` method."""
        y_mean, y_std = self.predict(x, verbose=verbose)
        return y_mean, y_std

    def predict(
        self, x: np.ndarray, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._windows is not None, "Windows are not initialized."

        # Split x into prediction windows
        pred_windows_ids = self._windows.create_prediction_windows_ids(x)

        y_mean, y_std = [], []
        for window_id in range(len(self._windows)):
            # Extract part of x in a given window
            start_id, end_id = pred_windows_ids[window_id]
            x_window = x[start_id : end_id + 1, :]

            y_mean_window, y_std_window = self.predict_window(
                x=x_window, window_id=window_id, verbose=verbose
            )
            y_mean.append(y_mean_window)
            y_std.append(y_std_window)

            if verbose:
                pprint("- Indices:", start_id, end_id, level=1)

        y_mean = np.hstack(y_mean)
        y_std = np.hstack(y_std)
        return y_mean, y_std

    def _build(self) -> None:
        assert self._windows is not None, "Windows are not initialized."

        x, y = self._windows.gather_data()
        assert len(x.shape) == 2, "Input x with shape {} is not 2D.".format(x.shape)
        assert len(y.shape) == 2, "Input y with shape {} is not 2D.".format(y.shape)

        self._nc.compute_normalization_values(x, y)

        for window in self._windows:
            window.model = copy.deepcopy(self._model)

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
    ) -> None:
        self._windows = create_windows(
            x,
            y,
            pred_window_width=self.pred_window_width,
            fit_window_width=self.fit_window_width,
            verbose=verbose,
        )
        self._build()
        self._fit(
            batch_size=batch_size,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_prints=n_prints,
            n_evals=n_evals,
            random_seed=random_seed,
            verbose=verbose,
        )

    def _fit(
        self,
        batch_size: int = 200,
        max_iter: int = 10000,
        learning_rate: float = 0.005,
        n_prints: int = 5,
        n_evals: int = 5,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        assert self._windows is not None, "Windows are not initialized."

        for window in self._windows:
            if verbose:
                print(str(window) + "\n")

            model = window.model

            # Normalize values
            x_window = self._nc.normalize_x(window.x)
            y_window = self._nc.normalize_y(window.y)

            x_inducing_window = (
                self._nc.normalize_x(window.x_inducing)
                if window.x_inducing is not None
                else None
            )
            x_val_window = (
                self._nc.normalize_x(window.x_val) if window.x_val is not None else None
            )

            # Train window model
            model.fit(
                x=x_window,
                y=y_window,
                x_inducing=x_inducing_window,
                batch_size=batch_size,
                max_iter=max_iter,
                learning_rate=learning_rate,
                n_prints=n_prints,
                x_val=x_val_window,
                n_evals=n_evals,
                random_seed=random_seed,
                verbose=verbose,
            )

    def fit_from_windows(
        self,
        windows: Windows,
        n_prints: int = 5,
        batch_size: int = 200,
        max_iter: int = 10000,
        learning_rate: float = 0.005,
        n_evals: int = 5,
        random_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self._windows = windows
        self._build()
        self._fit(
            batch_size=batch_size,
            max_iter=max_iter,
            learning_rate=learning_rate,
            n_prints=n_prints,
            n_evals=n_evals,
            random_seed=random_seed,
            verbose=verbose,
        )

    def predict_window(
        self, x: np.ndarray, window_id: int, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self._windows is not None, "Windows are not initialized."
        assert (
            0 <= window_id < len(self._windows)
        ), "Window index {} is out of bounds.".format(window_id)
        assert len(x.shape) == 2, "Input x with shape {} is not 2D.".format(x.shape)

        window = self._windows[window_id]
        model = window.model

        # Window prediction
        assert model is not None, "Window model is not initialized."

        # Normalize
        x = self._nc.normalize_x(x)

        # Predict
        y_mean_window, y_std_window = model(x)
        y_mean_window = np.reshape(y_mean_window, newshape=(-1, 1))
        y_std_window = np.reshape(y_std_window, newshape=(-1, 1))

        # Denormalize
        y_mean_window = self._nc.denormalize_y(y_mean_window)
        y_std_window = self._nc.denormalize_y(y_std_window, y_shift=0.0)

        if verbose:
            print(str(window) + "\n")
            pprint("- x_window:", x.shape, level=1)
            pprint("- y_mean:", y_mean_window.shape, level=1)
            pprint("- y_std:", y_std_window.shape, level=1)

        return y_mean_window.ravel(), y_std_window.ravel()
