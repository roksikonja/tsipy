"""
Plot utilities for visualizing signals, correction history and signals with
confidence intervals.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import style
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import norm

__all__ = [
    "COLORS",
    "plot_signals",
    "plot_signals_history",
    "plot_signals_and_confidence",
]

COLORS = list(mcolors.TABLEAU_COLORS)


def set_style(
    style_name: str = "seaborn",
    fig_size: Tuple[int, int] = (12, 6),
    font_size: int = 16,
    ax_font_size: int = 18,
    ticks_font_size: int = 16,
    title_font_size: int = 16,
    legend_font_size: int = 16,
    marker_type: str = "x",
    out_format: str = "png",
    latex: bool = False,
) -> None:
    """Set `pyplot` style parameters."""
    style.use(style_name)

    rc_params = {
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
    if latex:
        latex_rc_params = {
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
        }
        rc_params["savefig.format"] = "pdf"

        rc_params = {**rc_params, **latex_rc_params}

    mpl.rcParams.update(rc_params)


def configure_plot(
    ax: Axes,
    x_ticker: int = None,
    legend: str = None,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    x_label: str = None,
    y_label: str = None,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
) -> None:
    """Helper function for configuring axes parameters."""
    # pylint: disable=C0103
    if x_ticker:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))

    if legend:
        ax.legend(loc=legend)

    if x_lim:
        ax.set_xlim(*x_lim)

    if y_lim:
        ax.set_ylim(*y_lim)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if log_scale_x:
        ax.set_xscale("log")

    if log_scale_y:
        ax.set_yscale("log")


def plot_signals(
    signal_fourplets: List[
        Tuple[np.ndarray, np.ndarray, str, Dict],
    ],
    results_dir: Optional[str] = None,
    title: Optional[str] = None,
    tight_layout: bool = True,
    show: bool = False,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Helper function for plotting signals."""
    # pylint: disable=R0914
    fig, ax = plt.subplots()
    for signal_fourplet in signal_fourplets:
        x, y, label, kwargs_sig = signal_fourplet

        # Delete NaNs
        index_y_nn = ~np.isnan(y)
        x = x[index_y_nn]
        y = y[index_y_nn]

        ax.plot(x, y, label=label, **kwargs_sig)

    configure_plot(ax, **kwargs)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if results_dir is not None and title is not None:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax


def plot_signals_and_confidence(
    signal_fourplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    results_dir: Optional[str] = None,
    title: Optional[str] = None,
    confidence: float = 0.95,
    alpha: float = 0.5,
    tight_layout: bool = False,
    show: bool = False,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Helper function for plotting signal mean and confidence interval."""
    # pylint: disable=R0914

    # Computes confidence interval width for Normal(0, 1)
    factor = norm.ppf(1 / 2 + confidence / 2)  # 0.95 % -> 1.959963984540054

    fig, ax = plt.subplots()
    for signal_fourplet in signal_fourplets:
        x, y_mean, y_std, label = signal_fourplet

        ax.plot(x, y_mean, label=label)
        ax.fill_between(
            x,
            y_mean - factor * y_std,
            y_mean + factor * y_std,
            alpha=alpha,
            label=None,
        )

    configure_plot(ax, **kwargs)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if results_dir is not None and title is not None:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax


def plot_signals_history(
    x: np.ndarray,
    signals_history: List[List[Tuple[np.ndarray, str]]],
    results_dir: Optional[str] = None,
    title: Optional[str] = None,
    n_rows: int = 2,
    n_cols: int = 2,
    fig_size: Tuple[int, int] = (12, 6),
    tight_layout: bool = False,
    show: bool = False,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """Helper function for plotting degradation correction history."""
    # pylint: disable=R0914
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

    for i, signals in enumerate(signals_history):
        if n_rows == 1 and n_cols == 1:
            ax = axs
        elif n_rows == 1:
            ax = axs[i]
        elif n_cols == 1:
            ax = axs[i]
        else:
            col = i % n_cols
            row = i // n_rows
            ax = axs[row, col]

        for signal_pair in signals:
            y, label = signal_pair

            ax.plot(x, y, label=label)

        configure_plot(ax, **kwargs)

    if tight_layout:
        fig.tight_layout()

    if show:
        fig.show()

    if results_dir is not None and title is not None:
        fig.savefig(os.path.join(results_dir, title))

    return fig, axs


set_style()
