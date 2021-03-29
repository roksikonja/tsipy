import os
from typing import List, Dict, Union, Optional
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import style
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm

__all__ = [
    "Visualizer",
    "visualizer",
    "plot_signals",
    "plot_signals_history",
    "plot_signals_and_confidence",
]


class Visualizer:
    def __init__(
        self,
        style_name: str = "seaborn",
        fig_size: Tuple[int, int] = (12, 6),
        font_size: int = 16,
        ax_font_size: int = 18,
        ticks_font_size: int = 16,
        title_font_size: int = 16,
        legend_font_size: int = 16,
        marker_type: str = "x",
        out_format="pdf",
    ):
        style.use(style_name)

        rc_params = {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": "\n".join(
                [
                    "\\usepackage[utf8]{inputenc}",
                    "\\DeclareUnicodeCharacter{2212}{-}",
                ]
            ),
            "figure.figsize": fig_size,
            "font.size": font_size,
            "legend.fontsize": legend_font_size,
            "legend.title_fontsize": title_font_size,
            "axes.labelsize": ax_font_size,
            "xtick.labelsize": ticks_font_size,
            "ytick.labelsize": ticks_font_size,
            "scatter.marker": marker_type,
            "savefig.format": out_format,
        }

        mpl.rcParams.update(rc_params)


def configure_plot(
    ax: Axes,
    x_ticker=None,
    legend: str = None,
    y_lim: float = None,
    x_label: str = None,
    y_label: str = None,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
):
    if x_ticker:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))

    if legend:
        ax.legend(loc=legend)

    if y_lim:
        ax.set_ylim(y_lim)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if log_scale_x:
        ax.set_xscale("log")

    if log_scale_y:
        ax.set_yscale("log")


def plot_signals(
    signal_fiveplets: List[
        Union[
            Tuple[np.ndarray, np.ndarray, str, bool],
            Tuple[np.ndarray, np.ndarray, str, bool, Dict],
        ]
    ],
    results_dir: Optional[str] = None,
    title: Optional[str] = None,
    tight_layout: bool = True,
    show: bool = False,
    **kwargs,
) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    for signal_fiveplet in signal_fiveplets:
        if not signal_fiveplet:
            continue

        t = signal_fiveplet[0]
        x = signal_fiveplet[1]

        # Delete NaNs
        index_x_nn = ~np.isnan(x)
        t = t[index_x_nn]
        x = x[index_x_nn]

        label = signal_fiveplet[2]
        scatter = signal_fiveplet[3]
        if len(signal_fiveplet) == 5:
            kwargs_sig = signal_fiveplet[4]
        else:
            kwargs_sig = dict()

        if scatter:
            if "marker" not in kwargs_sig:
                kwargs_sig["marker"] = "x"
            if "color" not in kwargs_sig:
                kwargs_sig["color"] = "tab:red"

            ax.scatter(t, x, label=label, **kwargs_sig)
        else:
            ax.plot(t, x, label=label, **kwargs_sig)

    configure_plot(ax, **kwargs)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if results_dir is not None and title is not None:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax


def plot_signals_and_confidence(
    signal_fiveplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    results_dir: Optional[str] = None,
    title: Optional[str] = None,
    confidence: float = 0.95,
    alpha: float = 0.5,
    tight_layout: bool = False,
    show: bool = False,
    **kwargs,
) -> Tuple[Figure, Axes]:
    factor = norm.ppf(1 / 2 + confidence / 2)  # 0.95 % -> 1.959963984540054

    fig, ax = plt.subplots()
    for signal_fiveplet in signal_fiveplets:
        t = signal_fiveplet[0]
        x_mean = signal_fiveplet[1]
        x_std = signal_fiveplet[2]
        label = signal_fiveplet[3]

        ax.plot(t, x_mean, label=label)
        ax.fill_between(
            t,
            x_mean - factor * x_std,
            x_mean + factor * x_std,
            alpha=alpha,
            label=None,
        )

    configure_plot(ax, **kwargs)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if results_dir:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax


def plot_signals_history(
    t_m: np.ndarray,
    signals_history: List[List[Tuple]],
    results_dir: Optional[str] = None,
    title: Optional[str] = None,
    n_rows: int = 2,
    n_cols: int = 2,
    fig_size: Tuple[int, int] = (12, 6),
    tight_layout: bool = False,
    show: bool = False,
    **kwargs,
) -> Tuple[Figure, Axes]:
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

    for i, signals in enumerate(signals_history):
        col = i % n_cols
        row = i // n_rows

        ax = axs[row, col]

        for signal_triplet in signals:
            x = signal_triplet[0]
            label = signal_triplet[1]

            if len(signal_triplet) == 3:
                kwargs_sig = signal_triplet[2]
            else:
                kwargs_sig = dict()

            ax.plot(t_m, x, label=label, **kwargs_sig)

        configure_plot(ax, **kwargs)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if results_dir:
        fig.savefig(os.path.join(results_dir, title))

    return fig, axs


visualizer = Visualizer()
