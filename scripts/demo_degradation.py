import argparse
import os

import numpy as np

from tsipy.correction import (
    compute_exposure,
    correct_degradation,
    load_model,
    SignalGenerator,
)
from tsipy.utils import pprint, pprint_block
from tsipy_utils.data import make_dir
from tsipy_utils.visualizer import plot_signals, plot_signals_history


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", default="demo_degradation", type=str)

    # Degradation correction
    parser.add_argument("--degradation_model", "-m", default="mr", type=str)

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    pprint_block("Experiment", args.experiment_name)
    results_dir = make_dir(os.path.join("../results", args.experiment_name))

    signal_generator = SignalGenerator(
        add_noise=False,
        downsampling_rates=(0.99, 0.2),
    )
    data = signal_generator.data

    a = data["a"].values
    b = data["b"].values
    t = data["t"].values

    # Compute exposure
    e_a = compute_exposure(a)
    e_b = compute_exposure(b)
    e_a /= signal_generator.length
    e_b /= signal_generator.length
    data["e_a"] = e_a
    data["e_b"] = e_b

    # Channel measurements
    t_a_nn, a_nn, e_a_nn = signal_generator.get_signal_nn("a")
    t_b_nn, b_nn, e_b_nn = signal_generator.get_signal_nn("b")

    # Mutual measurements
    data_m = data[["t", "a", "b", "e_a", "e_b"]].dropna()
    t_m = data_m["t"].values
    a_m, b_m = data_m["a"].values, data_m["b"].values
    e_a_m, e_b_m = data_m["e_a"].values, data_m["e_b"].values

    pprint_block("Data", level=2)
    pprint("Time", level=0)
    pprint("- " + "t", t.shape, level=1)

    pprint("Signal", level=0)
    pprint("- a", a.shape, np.sum(~np.isnan(a)), level=1)
    pprint("- a_m", a_m.shape, level=1)
    pprint("- e_a", e_a_m.shape, level=1)

    pprint("Signal", level=0)
    pprint("- b", b.shape, np.sum(~np.isnan(b)), level=1)
    pprint("- b_m", b_m.shape, level=1)
    pprint("- e_b", e_b_m.shape, "\n", level=1)

    plot_signals(
        [
            (t_a_nn, a_nn, "$a$", {}),
            (t_b_nn, b_nn, "$b$", {}),
            (signal_generator.x, signal_generator.y, "$s$", {}),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        show=args.figure_show,
    )

    pprint_block("Degradation Correction", level=2)
    degradation_model = load_model(args.degradation_model)
    degradation_model.initial_fit(a_m, b_m, e_a_m)

    a_m_c, b_m_c, degradation_model, history = correct_degradation(
        t_m,
        a_m,
        e_a_m,
        b_m,
        e_b_m,
        model=degradation_model,
    )

    d_a_c = degradation_model(e_a_nn)
    d_b_c = degradation_model(e_b_nn)
    a_c_nn = np.divide(a_nn, d_a_c)
    b_c_nn = np.divide(b_nn, d_b_c)

    pprint_block("Results", level=2)

    plot_signals(
        [
            (t_m, a_m_c, "$a_c$", {}),
            (t_m, b_m_c, "$b_c$", {}),
            (signal_generator.x, signal_generator.y, "$s$", {}),
        ],
        results_dir=results_dir,
        title="signals_corrected",
        legend="upper right",
        show=args.figure_show,
    )

    plot_signals(
        [
            (t_a_nn, d_a_c, "$d_c(e_a(t))$", {}),
            (t_b_nn, d_b_c, "$d_c(e_b(t))$", {}),
            (
                t_a_nn,
                signal_generator.degradation_model(e_a_nn),
                "$d(e_a(t))$",
                {},
            ),
        ],
        results_dir=results_dir,
        title="degradation",
        legend="lower left",
        show=args.figure_show,
    )

    plot_signals_history(
        t_m,
        [
            [
                (signals.a, "$a_{}$".format(i)),
                (signals.b, "$b_{}$".format(i)),
                (
                    signal_generator.y[
                        np.logical_and(
                            signal_generator.get_indices_nn("a"),
                            signal_generator.get_indices_nn("b"),
                        )
                    ],
                    "$s$",
                ),
            ]
            for i, signals in enumerate(history[:4])
        ],
        results_dir=results_dir,
        title="correction_history",
        n_rows=2,
        n_cols=2,
        tight_layout=True,
        show=args.figure_show,
    )
