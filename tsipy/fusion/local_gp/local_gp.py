import copy

import numpy as np
import tensorflow as tf

from ...utils import normalize, denormalize, pformat, pprint_block


class LocalGPModel:
    def __init__(
        self,
        windows,
        model,
    ):
        self._model = model
        self.windows = windows

    def __call__(self, x, verbose=False):
        if verbose:
            pprint_block("Prediction", level=2)

        # Normalize
        x = normalize(x.copy(), self.windows.x_mean, self.windows.x_std)
        pred_windows_ids = self.windows.create_prediction_windows_ids(x)

        y_mean = np.zeros(shape=(x.shape[0],))
        y_std = np.zeros(shape=(x.shape[0],))
        for window, (start_id, end_id) in zip(self.windows.list, pred_windows_ids):
            model = window.model
            x_window = x[start_id : end_id + 1, :]

            # Window prediction
            y_window_mean, y_window_std = model(x_window)
            y_mean[start_id : end_id + 1] = y_window_mean
            y_std[start_id : end_id + 1] = y_window_std

            if verbose:
                x_str = pformat("    - x_window:", x_window.shape)
                ids_str = pformat("    - Indices:", start_id, end_id)
                range_str = pformat(
                    "    - Range:",
                    "{:.3f}, {:>8.3f}".format(x_window[0, 0], x_window[-1, 0]),
                )
                print(str(window) + "\n")
                print("\n".join([x_str, ids_str, range_str]))

        # Denormalize
        y_mean = denormalize(y_mean, self.windows.y_mean, self.windows.y_std)
        y_std = denormalize(y_std, 0.0, self.windows.y_std)
        return y_mean, y_std

    def _build_models(self):
        for window in self.windows.list:
            window.model = copy.deepcopy(self._model)
            window.model._compute_normalization_values(window.x, window.y)
            window.model._build_model(window.x, window.x_inducing)

    def fit(self, n_prints=5, verbose=False, **kwargs):
        if verbose:
            pprint_block("Training", level=2)

        self._build_models()

        for window in self.windows.list:
            print(str(window) + "\n")

            model = window.model

            # TF Dataset
            dataset = tf.data.Dataset.from_tensor_slices((window.x, window.y))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=window.x.shape[0])

            # Train
            model.train(
                dataset=dataset, n_prints=n_prints, x_val=window.x_val, **kwargs
            )
