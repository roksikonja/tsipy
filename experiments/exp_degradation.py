import numpy as np

import tsipy
from tsipy.correction import (
    ExposureMethod,
    compute_exposure,
    DegradationModel,
    CorrectionMethod,
    correct_degradation,
    SignalGenerator,
)
from utils import create_results_dir
from utils.visualizer import pprint, plot_signals, plot_signals_history


if __name__ == "__main__":
    results_dir = create_results_dir("../results", "exp-degradation")

    """
        Parameters
    """
    exposure_method = ExposureMethod.NUM_MEASUREMENTS
    correction_method = CorrectionMethod.CORRECT_ONE
    degradation_model_type = DegradationModel.SMR

    """
        Dataset
    """
    for i in range(5):
        np.random.seed(i)
        seed_str = "-" + str(i)

        t_field = "t"
        e_field = "e"

        a_field = "a"
        b_field = "b"

        t_a_field, t_b_field = t_field + "_a", t_field + "_b"
        e_a_field, e_b_field = e_field + "_a", e_field + "_b"

        signal_generator = SignalGenerator(
            length=50000,
            downsampling_a=0.99,
            downsampling_b=0.2,
            std_noise_a=0.0,
            std_noise_b=0.0,
            random_seed=i,
        )
        data = signal_generator.data

        pprint("\t- data", data.shape)
        print(data.head().to_string() + "\n")

        a = data[a_field].values
        b = data[b_field].values
        t = data[t_field].values

        pprint("\t- " + t_field, t.shape)
        pprint("\t- " + a_field, a.shape, np.sum(~np.isnan(a)))
        pprint("\t- " + b_field, b.shape, np.sum(~np.isnan(b)))

        # Compute exposure
        e_a = compute_exposure(a, method=exposure_method)
        e_b = compute_exposure(b, method=exposure_method)
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

        pprint("\t- " + t_a_field, t_a_nn.min(), t_a_nn.max())
        pprint("\t- " + t_b_field, t_b_nn.min(), t_b_nn.max())

        # Mutual measurements
        data_m = data[[t_field, a_field, b_field, e_a_field, e_b_field]].dropna()
        t_m = data_m[t_field].values
        a_m, b_m = data_m[a_field].values, data_m[b_field].values
        e_a_m, e_b_m = data_m[e_a_field].values, data_m[e_b_field].values

        pprint("\t- " + a_field, a_m.shape, np.sum(~np.isnan(a_m)))
        pprint("\t- " + b_field, b_m.shape, np.sum(~np.isnan(b_m)))
        pprint("\t- " + e_a_field, e_a_m.shape)
        pprint("\t- " + e_b_field, e_b_m.shape, "\n")

        fig, _ = plot_signals(
            [
                (t_a_nn, a_nn, r"$a$", False),
                (t_b_nn, b_nn, r"$b$", False),
                (signal_generator.t, signal_generator.s, r"$s$", False),
            ],
            results_dir=results_dir,
            title="signals" + seed_str,
            legend="upper right",
            tight_layout=True,
        )
        fig.show()

        """
            Degradation correction
        """
        degradation_model = tsipy.correction.load_model(degradation_model_type)
        degradation_model.initial_fit(a_m, b_m, e_a_m)

        a_m_c, b_m_c, degradation_model, history = correct_degradation(
            t_m,
            a_m,
            e_a_m,
            b_m,
            e_b_m,
            model=degradation_model,
            method=correction_method,
        )

        d_a_c = degradation_model(e_a_nn)
        d_b_c = degradation_model(e_b_nn)
        a_c_nn = np.divide(a_nn, d_a_c)
        b_c_nn = np.divide(b_nn, d_b_c)

        fig, _ = plot_signals(
            [
                (t_m, a_m_c, r"$a_c$", False),
                (t_m, b_m_c, r"$b_c$", False),
                (signal_generator.t, signal_generator.s, r"$s$", False),
            ],
            results_dir=results_dir,
            title="signals_corrected" + seed_str,
            legend="upper right",
        )
        fig.show()

        fig, _ = plot_signals(
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
        )
        fig.show()

        fig, _ = plot_signals_history(
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
            tight_layout=True,
        )
        fig.show()
