import numpy as np

from ..models_gp import NormalizationClippingMixin
from ...utils import (
    normalize,
    clipping_indices,
    get_window_indices,
    pformat,
    pprint,
    pprint_block,
)


class Window:
    def __init__(
        self,
        x_pred_start,
        x_pred_end,
        x_fit_start,
        x_fit_end,
        data_start_id,
        data_end_id,
        x,
        y,
    ):
        self.x_pred_start = x_pred_start
        self.x_pred_end = x_pred_end

        self.x_fit_start = x_fit_start
        self.x_fit_end = x_fit_end

        # Data
        self.data_start_id = data_start_id
        self.data_end_id = data_end_id
        self.x = x
        self.y = y

        # GP model parameters
        self.model = None
        self.x_inducing = None
        self.x_val = None

    def __str__(self):
        pred_str = pformat(
            "    - Prediction:",
            "{:.3f}, {:>8.3f}".format(self.x_pred_start, self.x_pred_end),
        )
        fit_str = pformat(
            "    - Training:",
            "{:.3f}, {:>8.3f}".format(self.x_fit_start, self.x_fit_end),
        )
        data_str = pformat(
            "    - Data indices:",
            "{}, {:>8}".format(self.data_start_id, self.data_end_id),
        )
        return "\n".join(["Window", pred_str, fit_str, data_str])


class Windows(NormalizationClippingMixin):
    def __init__(self, x, y, normalization=True, clipping=True):
        super(Windows, self).__init__(normalization=normalization, clipping=clipping)

        self.list = []

        self._compute_normalization_values(x, y)
        x = normalize(x.copy(), self.x_mean, self.x_std)
        y = normalize(y.copy(), self.y_mean, self.y_std)

        if self.clipping:
            clip_indices = clipping_indices(y)
            x, y = x[clip_indices, :], y[clip_indices]

        self.x = np.atleast_2d(x)
        self.y = y.reshape(-1, 1)

    def __str__(self):
        strings = ["Windows:"]
        for i, window in enumerate(self.list):
            window_str = [pformat("    Window:", i)]
            window_str += [(4 * " ") + s for s in str(window).split("\n")[1:]]
            strings.append("\n".join(window_str))

        return "\n".join(strings)

    def create_prediction_windows_ids(self, x, verbose=False):
        if verbose:
            pprint_block("Prediction Windows Creation", level=2)

        pred_windows = []
        start_id = 0
        for window in self.list:
            _, end_id = get_window_indices(
                x[:, 0], x_start=window.x_pred_start, x_end=window.x_pred_end
            )

            pred_windows.append((start_id, end_id))

            if verbose:
                ids_str = pformat("    - Indices:", start_id, end_id)
                range_str = pformat(
                    "    - Range:",
                    "{:.3f}, {:>8.3f}".format(x[start_id, 0], x[end_id, 0]),
                )
                print("\n" + str(window) + "\n")
                print("\n".join([ids_str, range_str]))

            start_id = end_id + 1

        return pred_windows


def create_prediction_windows(x, pred_window, verbose=False):
    """
    Returns: A list of (x_start, x_end, x_mid) triplets corresponding to the prediction window bounds and
        window center.

    """
    x_min, x_max = x[:, 0].min(), x[:, 0].max()

    pred_windows = []
    x_pred_start = -float("inf")
    for x_pred_end in np.arange(x_min, x_max, pred_window)[1:]:
        if x_pred_start > -float("inf"):
            x_pred_mid = x_pred_start + pred_window / 2.0
        else:
            x_pred_mid = x_min + pred_window / 2.0

        pred_windows.append((x_pred_start, x_pred_end, x_pred_mid))
        x_pred_start = x_pred_end

    x_pred_mid = x_pred_start + pred_window / 2.0
    pred_windows.append((x_pred_start, float("inf"), x_pred_mid))

    if verbose:
        pprint("Prediction windows:")
        for start, end, mid in pred_windows:
            pprint("    Window:")
            pprint("        - Range:", start, end)
            pprint("        - Center:", mid)

    return pred_windows


def create_fit_windows(x, fit_window, pred_windows, verbose=False):
    x_min, x_max = x[:, 0].min(), x[:, 0].max()

    fit_windows = []
    for x_pred_start, x_pred_end, x_pred_mid in pred_windows:
        x_fit_start = max(x_min, x_pred_mid - fit_window / 2.0)
        x_fit_end = min(x_max, x_pred_mid + fit_window / 2.0)
        fit_windows.append((x_fit_start, x_fit_end))

    if verbose:
        pprint("Fit windows:")
        for start, end in fit_windows:
            pprint("    Window:")
            pprint("        - Range:", start, end)

    return fit_windows


def create_windows(x, y, pred_window, fit_window, verbose=False, **kwargs):
    if verbose:
        pprint_block("Windows Creation", level=2)

    windows = Windows(x, y, **kwargs)

    pred_windows = create_prediction_windows(windows.x, pred_window, verbose=verbose)
    fit_windows = create_fit_windows(
        windows.x, fit_window, pred_windows, verbose=verbose
    )

    for pred_window, fit_window in zip(pred_windows, fit_windows):
        start_id, end_id = get_window_indices(
            windows.x[:, 0], x_start=fit_window[0], x_end=fit_window[1]
        )

        x_window = windows.x[start_id : end_id + 1, :]
        y_window = windows.y[start_id : end_id + 1, :]

        window = Window(
            x_pred_start=pred_window[0],
            x_pred_end=pred_window[1],
            x_fit_start=fit_window[0],
            x_fit_end=fit_window[1],
            data_start_id=start_id,
            data_end_id=end_id,
            x=x_window,
            y=y_window,
        )
        windows.list.append(window)

    if verbose:
        print(windows)

    return windows
