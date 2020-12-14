import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import norm

from ..constants import Constants as Const
from ..data import transform_time_to_unit


def plot_signals(
        signal_fourplets,
        results_dir,
        title,
        x_ticker=None,
        legend=None,
        y_lim=None,
        x_label=None,
        y_label=None,
):
    fig, ax = plt.subplots()
    for signal_fourplet in signal_fourplets:
        if not signal_fourplet:
            continue

        t = signal_fourplet[0]
        x = signal_fourplet[1]

        # Delete NaNs
        index_x_nn = ~np.isnan(x)
        t = t[index_x_nn]
        x = x[index_x_nn]

        label = signal_fourplet[2]
        scatter = signal_fourplet[3]

        t = transform_time_to_unit(t, x_label=x_label)

        if scatter:
            ax.scatter(t, x, label=label, marker="x", color="tab:red")
        else:
            ax.plot(t, x, label=label, lw=Const.LW)

    if x_ticker:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_ticker))

    if legend:
        ax.legend(loc=legend)
    else:
        ax.legend()

    if y_lim:
        ax.set_ylim(y_lim)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if results_dir:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax


def plot_signals_mean_std_precompute(
        signal_fourplets,
        results_dir,
        title,
        x_ticker=None,
        legend=None,
        y_lim=None,
        x_label=None,
        y_label=None,
        confidence=0.95,
        alpha=0.5,
):
    factor = norm.ppf(1 / 2 + confidence / 2)  # 0.95 % -> 1.959963984540054

    fig, ax = plt.subplots()
    for signal_fourplet in signal_fourplets:
        t = signal_fourplet[0]
        x_mean = signal_fourplet[1]
        x_std = signal_fourplet[2]
        label = signal_fourplet[3]

        t = transform_time_to_unit(t, x_label=x_label)

        ax.plot(t, x_mean, label=label)
        ax.fill_between(
            t,
            x_mean - factor * x_std,
            x_mean + factor * x_std,
            alpha=alpha,
            label=label,
        )

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

    if results_dir:
        fig.savefig(os.path.join(results_dir, title))

    return fig, ax
