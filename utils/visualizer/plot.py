import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import norm

from ..constants import Constants as Const


def configure_plot(
    ax,
    x_ticker=None,
    legend=None,
    y_lim=None,
    x_label=None,
    y_label=None,
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


def plot_signals(
    signal_fiveplets,
    results_dir,
    title,
    tight_layout=False,
    **kwargs,
):
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
            if "lw" not in kwargs_sig:
                kwargs_sig["lw"] = Const.LW
            ax.plot(t, x, label=label, **kwargs_sig)

    configure_plot(ax, **kwargs)

    if tight_layout:
        fig.tight_layout()

    if results_dir:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax


def plot_signals_and_confidence(
    signal_fiveplets,
    results_dir,
    title,
    confidence=0.95,
    alpha=0.5,
    tight_layout=False,
    **kwargs,
):
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

    if results_dir:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax


def plot_signals_history(
    t_m,
    signals_history,
    results_dir,
    title,
    n_rows=2,
    n_cols=2,
    fig_size=(12, 6),
    tight_layout=False,
    **kwargs,
):

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

    if results_dir:
        fig.savefig(os.path.join(results_dir, title))

    return fig, axs
