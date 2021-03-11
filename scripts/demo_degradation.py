import argparse

import numpy as np

import tsipy.correction
from tsipy.utils import pprint, pformat, pprint_block
from utils.data import create_results_dir
from utils.visualizer import plot_signals, plot_signals_history


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="demo_degradation", type=str)

    # Degradation correction
    parser.add_argument("--degradation_model", default="mr", type=str)

    # Visualize
    parser.add_argument("-figure_show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    results_dir = create_results_dir("../results", args.experiment_name)

    for i in range(5):
        pprint_block(pformat("Experiment:", i))
        np.random.seed(i)
        seed_str = "-" + str(i)

        t_field = "t"
        e_field = "e"

        a_field = "a"
        b_field = "b"

        t_a_field, t_b_field = t_field + "_a", t_field + "_b"
        e_a_field, e_b_field = e_field + "_a", e_field + "_b"

        signal_generator = tsipy.correction.SignalGenerator(
            add_noise=False,
            downsampling_a=0.99,
            downsampling_b=0.2,
            random_seed=i,
        )
        data = signal_generator.data

        a = data[a_field].values
        b = data[b_field].values
        t = data[t_field].values

        print(data, "\n")
        pprint("- data", data.shape)
        pprint("- " + t_field, t.shape)
        pprint("- " + a_field, a.shape, np.sum(~np.isnan(a)))
        pprint("- " + b_field, b.shape, np.sum(~np.isnan(b)))

        # Compute exposure
        e_a = tsipy.correction.compute_exposure(a)
        e_b = tsipy.correction.compute_exposure(b)
        max_e = max(np.max(e_a), np.max(e_b))
        e_a, e_b = e_a / max_e, e_b / max_e
        data[e_a_field] = e_a
        data[e_b_field] = e_b

        # Channel measurements
        data_a = data[[t_field, a_field, e_a_field]].dropna()
        data_b = data[[t_field, b_field, e_b_field]].dropna()

        t_a_nn, t_b_nn = data_a[t_field].values, data_b[t_field].values
        a_nn, b_nn = data_a[a_field].values, data_b[b_field].values
        e_a_nn, e_b_nn = data_a[e_a_field].values, data_b[e_b_field].values

        pprint("- " + t_a_field, t_a_nn.min(), t_a_nn.max())
        pprint("- " + t_b_field, t_b_nn.min(), t_b_nn.max())

        # Mutual measurements
        data_m = data[[t_field, a_field, b_field, e_a_field, e_b_field]].dropna()
        t_m = data_m[t_field].values
        a_m, b_m = data_m[a_field].values, data_m[b_field].values
        e_a_m, e_b_m = data_m[e_a_field].values, data_m[e_b_field].values

        pprint("- " + a_field, a_m.shape, np.sum(~np.isnan(a_m)))
        pprint("- " + b_field, b_m.shape, np.sum(~np.isnan(b_m)))
        pprint("- " + e_a_field, e_a_m.shape)
        pprint("- " + e_b_field, e_b_m.shape, "\n")

        plot_signals(
            [
                (t_a_nn, a_nn, r"$a$", False),
                (t_b_nn, b_nn, r"$b$", False),
                (signal_generator.t, signal_generator.s, r"$s$", False),
            ],
            results_dir=results_dir,
            title="signals" + seed_str,
            legend="upper right",
            show=args.figure_show,
        )

        """
            Degradation correction
        """
        pprint_block("Degradation Correction", level=2)
        degradation_model = tsipy.correction.load_model(args.degradation_model)
        degradation_model.initial_fit(a_m, b_m, e_a_m)

        a_m_c, b_m_c, degradation_model, history = tsipy.correction.correct_degradation(
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

        """
            Results
        """
        pprint_block("Results", level=2)

        plot_signals(
            [
                (t_m, a_m_c, r"$a_c$", False),
                (t_m, b_m_c, r"$b_c$", False),
                (signal_generator.t, signal_generator.s, r"$s$", False),
            ],
            results_dir=results_dir,
            title="signals_corrected" + seed_str,
            legend="upper right",
            show=args.figure_show,
        )

        plot_signals(
            [
                (t_a_nn, d_a_c, r"$d_c(e_a(t))$", False),
                (t_b_nn, d_b_c, r"$d_c(e_b(t))$", False),
                (
                    t_a_nn,
                    signal_generator.degradation_model(
                        e_a[signal_generator.t_a_indices]
                    ),
                    r"$d(e_a(t))$",
                    False,
                ),
            ],
            results_dir=results_dir,
            title="degradation" + seed_str,
            legend="lower left",
            show=args.figure_show,
        )

        plot_signals_history(
            t_m,
            [
                [
                    (signals[0], r"$a_{}$".format(i)),
                    (signals[1], r"$b_{}$".format(i)),
                    (
                        signal_generator.s[
                            np.logical_and(
                                signal_generator.t_a_indices,
                                signal_generator.t_b_indices,
                            )
                        ],
                        r"$s$",
                    ),
                ]
                for i, signals in enumerate(history[:4])
            ],
            results_dir,
            title="correction-history" + seed_str,
            n_rows=2,
            n_cols=2,
            show=args.figure_show,
        )
