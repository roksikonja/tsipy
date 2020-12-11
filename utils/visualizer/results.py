import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from ..constants import Constants as Const
from ..data import mission_day_to_year, downsample_signal


def plot_signals(
    signal_fourplets,
    results_dir,
    title,
    x_ticker=None,
    legend=None,
    y_lim=None,
    x_label=None,
    y_label=None,
    max_points=1e6,
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

        if x_label == Const.YEAR_UNIT:
            t = np.array(list(map(mission_day_to_year, t)))

        if x.shape[0] > max_points:
            downsampling_factor = int(np.floor(float(x.shape[0]) / float(max_points)))
            t = downsample_signal(t, downsampling_factor)
            x = downsample_signal(x, downsampling_factor)

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

    fig.show()
    if results_dir:
        fig.savefig(os.path.join(results_dir, title))
