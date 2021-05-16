"""
A script for computing the 40-year TSI composite.
"""
import argparse
import datetime
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.signal

from tsipy.utils import (
    COLORS,
    make_dir,
    plot_signals,
    pprint,
    pprint_block,
    transform_time_to_unit,
)


def parse_arguments():
    """Parses command line arguments specifying processing method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", default="exp_tsi_40", type=str)
    parser.add_argument(
        "--dataset_name",
        "-d",
        default="acrim",
        type=str,
    )

    # Fusion Model
    parser.add_argument("--fusion_model", "-m", default="svgp", type=str)

    # Preprocess
    parser.add_argument("--normalization", "-n", action="store_false")
    parser.add_argument("--clipping", "-c", action="store_false")

    # SVGP
    parser.add_argument("--num_inducing_pts", "-n_ind_pts", default=100, type=int)
    parser.add_argument("--max_iter", default=1000, type=int)

    # Local GP
    parser.add_argument(
        "--pred_window",
        "-p_w",
        default=1.0,
        type=float,
        help="Width of prediction window in years.",
    )
    parser.add_argument(
        "--fit_window",
        "-f_w",
        default=3.0,
        type=float,
        help="Width of training window in years.",
    )

    # Visualize
    parser.add_argument("-figure_show", action="store_false")
    return parser.parse_args()


def load_dataset(dataset_path_: str, columns_: Dict[int, str]) -> pd.DataFrame:
    data_ = pd.read_csv(
        dataset_path_,
        delimiter=" ",
        header=None,
    ).rename(columns=columns_)

    data_["t_org"] = data_["t"].values.copy()
    data_["t"] = transform_time_to_unit(
        data_["t"] - data_["t"][0],
        t_label="year",
        start=datetime.datetime(1980, 1, 1),
    )
    return data_


if __name__ == "__main__":
    args = parse_arguments()

    datasets_path = "../data/tsi_40"
    datasets = dict()
    for dataset, columns in [
        ("ACRIM1_HF_full2.txt", {0: "t", 1: "ACRIM1", 2: "HF"}),
        ("ACRIM2_ERBS_full.txt", {0: "t", 1: "ACRIM2", 2: "ERBS"}),
        ("MultiSatellites_HF_ERBSmultiply1.txt", {0: "t", 1: "HF", 2: "ERBS"}),
        ("VIRGOFused_ACRIM2_ERBS.txt", {0: "t", 1: "VIRGO", 2: "ACRIM2", 3: "ERBS"}),
        ("VIRGOFused_ACRIM3_ERBS.txt", {0: "t", 1: "VIRGO", 2: "ACRIM3", 3: "ERBS"}),
        ("VIRGOFused_ACRIM3_TIM0.txt", {0: "t", 1: "VIRGO", 2: "ACRIM3", 3: "TIM"}),
        ("VIRGOFused_TIM.txt", {0: "t", 1: "VIRGO", 2: "TIM"}),
        ("VIRGOFused_TIM_PREMOSFused.txt", {0: "t", 1: "VIRGO", 2: "TIM", 3: "PREMOS"}),
        ("VIRGOFused_TISIS.txt", {0: "t", 1: "VIRGO", 2: "TISIS"}),
    ]:
        pprint_block("Dataset", dataset)

        dataset_path = os.path.join(datasets_path, dataset)
        dataset = os.path.splitext(dataset)[0]

        data = load_dataset(dataset_path, columns)
        datasets[dataset] = data
        print(data.head(5))

    pprint_block("Experiment", args.experiment_name)
    results_dir = make_dir(os.path.join("../results", args.experiment_name))

    signals = defaultdict(list)
    for dataset, data in datasets.items():
        pprint_block(f"Dataset: {dataset}", level=1)

        t: np.ndarray = data["t"]

        signal_fourplets = []
        psd_fourplets = []
        for signal_name in data.columns:
            if signal_name in ["t", "t_org"]:
                continue

            signal: np.ndarray = data[signal_name]

            signal_fourplet: Tuple[np.ndarray, np.ndarray, str, Dict] = (
                t,
                signal,
                signal_name,
                {},
            )
            signal_fourplets.append(signal_fourplet)

            n_per_seg = min(1024, signal.size)
            freqs, psd = scipy.signal.welch(signal, fs=1.0, nperseg=n_per_seg)

            psd_fourplet: Tuple[np.ndarray, np.ndarray, str, Dict] = (
                freqs,
                psd,
                signal_name,
                {},
            )
            psd_fourplets.append(psd_fourplet)

            signals[signal_name].append((t, signal, dataset))

        plot_signals(
            signal_fourplets,
            results_dir=results_dir,
            title=f"signals_{dataset}",
            legend="upper right",
            x_ticker=1,
            show=args.figure_show,
        )

        plot_signals(
            psd_fourplets,
            results_dir=results_dir,
            title=f"signals_psd_{dataset}",
            legend="upper right",
            log_scale_x=True,
            show=args.figure_show,
        )

    pprint_block("Visualizing Signals per Dataset")
    signal_fourplets_per_dataset = []
    for dataset_id, (dataset, data) in enumerate(datasets.items()):
        pprint_block(f"Dataset: {dataset}", level=1)

        t = data["t"]
        for signal_name in data.columns:
            if signal_name in ["t", "t_org"]:
                continue

            signal = data[signal_name]
            signal_fourplet = (t, signal, signal_name, {"c": COLORS[dataset_id]})
            signal_fourplets_per_dataset.append(signal_fourplet)

    plot_signals(
        signal_fourplets_per_dataset,
        results_dir=results_dir,
        title="signals_per_datasets",
        show=args.figure_show,
    )

    pprint_block("Visualizing Signals per Signal Name")
    signal_fourplets_per_name = []
    for signal_id, (signal_name, signal_name_triplets) in enumerate(signals.items()):
        pprint("Signal", signal_name, level=0)
        for t, signal, dataset in signal_name_triplets:
            pprint("- Dataset", dataset, level=1)
            pprint(f"- t_{signal_name}", t.shape, level=2)
            pprint(f"- {signal_name}", signal.shape, level=2)

            signal_fourplet = (t, signal, dataset, {"c": COLORS[signal_id]})
            signal_fourplets_per_name.append(signal_fourplet)

    plot_signals(
        signal_fourplets_per_name,
        results_dir=results_dir,
        title="signals_per_signal_name",
        show=args.figure_show,
    )
