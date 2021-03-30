from typing import List, Iterator, Optional, Tuple

import numpy as np

from ..core import FusionModel
from ...utils import (
    is_sorted,
    get_window_indices,
    pformat,
    pprint,
    pprint_block,
)


class Window:
    def __init__(
        self,
        x_pred_start: float,
        x_pred_end: float,
        x_pred_mid: float,
        x_fit_start: float,
        x_fit_end: float,
        data_start_id: int,
        data_end_id: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.x_pred_start = x_pred_start
        self.x_pred_end = x_pred_end
        self.x_pred_mid = x_pred_mid

        self.x_fit_start = x_fit_start
        self.x_fit_end = x_fit_end

        # Data
        self.data_start_id = data_start_id
        self.data_end_id = data_end_id
        self.x = x
        self.y = y

        # GP model parameters
        self.model: Optional[FusionModel] = None
        self.x_inducing: Optional[np.ndarray] = None
        self.x_val: Optional[np.ndarray] = None

    def __str__(self) -> str:
        pred_str = pformat(
            "- Prediction:",
            "{:.3f}, {:>8.3f}".format(self.x_pred_start, self.x_pred_end),
            level=1,
        )
        fit_str = pformat(
            "- Training:",
            "{:.3f}, {:>8.3f}".format(self.x_fit_start, self.x_fit_end),
            level=1,
        )
        data_str = pformat(
            "- Data indices:",
            "{}, {:>8}".format(self.data_start_id, self.data_end_id),
            level=1,
        )
        return "\n".join(["Window", pred_str, fit_str, data_str])


class Windows:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self._list: List[Window] = []

        self.x: np.ndarray = np.atleast_2d(x)
        self.y: np.ndarray = y.reshape(-1, 1)

    def __str__(self) -> str:
        strings = ["Windows:"]
        for i, window in enumerate(self):
            window_str = [pformat("Window:", i, level=1)]
            window_str += [(4 * " ") + s for s in str(window).split("\n")[1:]]
            strings.append("\n".join(window_str))

        return "\n".join(strings)

    def __iter__(self) -> Iterator:
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, key: int) -> Window:
        return self._list[key]

    def add_window(self, window: Window) -> None:
        self._list.append(window)

    def gather_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        start_id = 0

        for i, window in enumerate(self):
            start_id = max(start_id, window.data_start_id)
            end_id = window.data_end_id + 1

            if start_id < end_id:
                w_start_id = start_id - window.data_start_id
                w_end_id = end_id - window.data_start_id

                x.append(window.x[w_start_id:w_end_id])
                y.append(window.y[w_start_id:w_end_id])

                start_id = end_id

        x = np.vstack(x)
        y = np.vstack(y)
        return x, y

    def create_prediction_windows_ids(
        self, x: np.ndarray, verbose: bool = False
    ) -> List[Tuple[int, int]]:
        if verbose:
            pprint_block("Prediction Windows Creation", level=2)

        pred_windows = []
        start_id = 0
        for window in self:
            _, end_id = get_window_indices(
                x[:, 0], x_start=window.x_pred_start, x_end=window.x_pred_end
            )

            pred_windows.append((start_id, end_id))

            if verbose:
                ids_str = pformat("- Indices:", start_id, end_id, level=1)
                range_str = pformat(
                    "- Range:",
                    "{:.3f}, {:>8.3f}".format(x[start_id, 0], x[end_id, 0]),
                    level=1,
                )
                print("\n" + str(window) + "\n")
                print("\n".join([ids_str, range_str]))

            start_id = end_id + 1

        return pred_windows


def create_prediction_windows(
    x: np.ndarray, pred_window_width: float, verbose: bool = False
) -> List[Tuple[float, float, float]]:
    """
    Returns: A list of (x_start, x_end, x_mid) triplets corresponding to the prediction window bounds and
        window center.

    """
    x_min, x_max = x[:, 0].min(), x[:, 0].max()

    pred_windows = []
    x_pred_start = -np.infty
    for x_pred_end in np.arange(x_min, x_max, pred_window_width)[1:]:
        if x_pred_start > -np.infty:
            x_pred_mid = x_pred_start + pred_window_width / 2.0
        else:
            x_pred_mid = x_min + pred_window_width / 2.0

        pred_windows.append((x_pred_start, x_pred_end, x_pred_mid))
        x_pred_start = x_pred_end

    # If statement is necessary when there is only one window
    x_pred_mid = x_pred_start if x_pred_start > -np.infty else x_min
    x_pred_mid += pred_window_width / 2.0

    pred_windows.append((x_pred_start, np.infty, x_pred_mid))

    if verbose:
        pprint("Prediction windows:")
        for start, end, mid in pred_windows:
            pprint("Window:", level=1)
            pprint("- Range:", "{:.3f}, {:>8.3f}".format(start, end), level=2)
            pprint("- Center:", "{:.3f}".format(mid), level=2)

    return pred_windows


def create_fit_windows(
    x: np.ndarray,
    fit_window_width: float,
    pred_windows: List[Tuple[float, float, float]],
    verbose: bool = False,
) -> List[Tuple[float, float]]:
    x_min, x_max = x[:, 0].min(), x[:, 0].max()

    fit_windows = []
    for x_pred_start, x_pred_end, x_pred_mid in pred_windows:
        x_fit_start = max(x_min, x_pred_mid - fit_window_width / 2.0)
        x_fit_end = min(x_max, x_pred_mid + fit_window_width / 2.0)
        fit_windows.append((x_fit_start, x_fit_end))

    if verbose:
        pprint("Fit windows:")
        for start, end in fit_windows:
            pprint("Window:", level=1)
            pprint("- Range:", "{:.3f}, {:>8.3f}".format(start, end), level=2)

    return fit_windows


def create_windows(
    x: np.ndarray,
    y: np.ndarray,
    pred_window_width: float,
    fit_window_width: float,
    verbose: bool = False,
) -> Windows:
    assert is_sorted(x[:, 0]), "Input array x is not sorted in dimension 0."
    assert (
        pred_window_width <= fit_window_width
    ), "Prediction window {} is wider than training window {}".format(
        pred_window_width, fit_window_width
    )

    if verbose:
        pprint_block("Data", level=2)
        pprint("x:", x.shape, level=1)
        pprint(
            "- Range:", "{:.3f}, {:>8.3f}".format(x[:, 0].min(), x[:, 0].max()), level=2
        )
        pprint("y:", y.shape, level=1)
        pprint(
            "- Range:", "{:.3f}, {:>8.3f}".format(y[:, 0].min(), y[:, 0].max()), level=2
        )

    if verbose:
        pprint_block("Windows Creation", level=2)

    windows = Windows(x, y)

    pred_windows = create_prediction_windows(windows.x, pred_window_width)
    fit_windows = create_fit_windows(windows.x, fit_window_width, pred_windows)

    for pred_window, fit_window in zip(pred_windows, fit_windows):
        start_id, end_id = get_window_indices(
            windows.x[:, 0], x_start=fit_window[0], x_end=fit_window[1]
        )

        x_window = windows.x[start_id : end_id + 1, :].copy()
        y_window = windows.y[start_id : end_id + 1, :].copy()

        window = Window(
            x_pred_start=pred_window[0],
            x_pred_end=pred_window[1],
            x_pred_mid=pred_window[2],
            x_fit_start=fit_window[0],
            x_fit_end=fit_window[1],
            data_start_id=start_id,
            data_end_id=end_id,
            x=x_window,
            y=y_window,
        )
        windows.add_window(window)

    if verbose:
        print(windows)

    return windows
