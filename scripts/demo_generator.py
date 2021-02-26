import os

import gpflow as gpf
import numpy as np
import tensorflow as tf

import tsipy.correction
from tsipy.correction import (
    ExposureMethod,
    compute_exposure,
    DegradationModel,
    CorrectionMethod,
    correct_degradation,
    SignalGenerator,
)
from tsipy.fusion import (
    SVGPModel,
    MultiWhiteKernel,
    build_sensor_labels,
    build_output_labels,
    concatenate_labels,
)
from utils import Constants as Const
from utils.data import create_results_dir
from utils.visualizer import (
    pprint,
    plot_signals,
    plot_signals_and_confidence,
    plot_signals_history,
)

if __name__ == "__main__":
    results_dir = create_results_dir("../results", "generator")

    """
        Parameters
    """
    exposure_method = ExposureMethod.NUM_MEASUREMENTS
    correction_method = CorrectionMethod.CORRECT_ONE
    degradation_model = DegradationModel.SMR

    """
        Dataset
    """
    np.random.seed(Const.RANDOM_SEED)
    tf.random.set_seed(Const.RANDOM_SEED)

    t_field = "t"
    e_field = "e"

    a_field = "a"
    b_field = "b"

    t_a_field, t_b_field = t_field + "_a", t_field + "_b"
    e_a_field, e_b_field = e_field + "_a", e_field + "_b"

    # Generate Brownian motion signal
    signal_generator = SignalGenerator()
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
        title="signals",
        legend="upper right",
        tight_layout=True,
    )
    fig.show()

    """
        Degradation correction
    """
    degradation_model = tsipy.correction.load_model(degradation_model)
    degradation_model.initial_fit(a_m, b_m, e_a_m)

    a_m_c, b_m_c, degradation_model, history = correct_degradation(
        t_m,
        a_m,
        e_a_m,
        b_m,
        e_b_m,
        model=degradation_model,
        method=correction_method,
        verbose=True,
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
        title="signals_corrected",
        legend="upper right",
    )
    fig.show()

    fig, _ = plot_signals(
        [
            (t_a_nn, d_a_c, r"$d_c(e_a(t))$", False),
            (t_b_nn, d_b_c, r"$d_c(e_b(t))$", False),
            (
                t_a_nn,
                signal_generator.degradation_model(e_a[signal_generator.t_a_indices]),
                r"$d(e_a(t))$",
                False,
            ),
        ],
        results_dir=results_dir,
        title="degradation",
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
                            signal_generator.t_a_indices, signal_generator.t_b_indices
                        )
                    ],
                    r"$s$",
                ),
            ]
            for i, signals in enumerate(history[:4])
        ],
        results_dir,
        title="correction-history",
        n_rows=2,
        n_cols=2,
        tight_layout=True,
    )
    fig.show()

    """
        Data fusion
    """
    gpf.config.set_default_float(np.float64)
    np.random.seed(Const.RANDOM_SEED)
    tf.random.set_seed(Const.RANDOM_SEED)

    pprint("\t- t_a_nn", t_a_nn.shape)
    pprint("\t- t_b_nn", t_b_nn.shape)
    pprint("\t- a_c_nn", a_c_nn.shape)
    pprint("\t- b_c_nn", b_c_nn.shape)

    labels, t_labels = build_sensor_labels((t_a_nn, t_b_nn))
    s = np.hstack((a_c_nn, b_c_nn))
    t = np.hstack((t_a_nn, t_b_nn))
    t = concatenate_labels(t, t_labels)

    pprint("\t- labels", labels)
    pprint("\t- t_labels", t_labels.shape)
    pprint("\t- t", t.shape)
    pprint("\t- s", s.shape)

    t_out = signal_generator.t
    t_out_labels = build_output_labels(t_out)
    t_out = concatenate_labels(t_out, t_out_labels)

    pprint("\t- t_out_labels", t_out_labels.shape)
    pprint("\t- t_out", t_out.shape)

    """
        Kernel
    """
    # Signal kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])  # Kernel for time dimension

    # Noise kernel
    white_kernel = MultiWhiteKernel(
        labels=labels, active_dims=[1]
    )  # Kernel for sensor dimension

    # Kernel composite
    kernel = matern_kernel + white_kernel

    """
        Gaussian Process Model
    """
    fusion_model = tsipy.fusion.models.SVGPModel(kernel=kernel, num_inducing_pts=250)

    # Train
    fusion_model.fit(t, s, max_iter=2500, verbose=True, x_val=t_out, n_evals=5)

    # Predict
    s_out_mean, s_out_std = fusion_model(t_out)
    t_out = t_out[:, 0]

    pprint("\t- t_out", t_out.shape)
    pprint("\t- s_out_mean", s_out_mean.shape)
    pprint("\t- s_out_std", s_out_std.shape)

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
    )
    fig.show()
    ax.scatter(
        t_a_nn,
        a_c_nn,
        label="$a_c$",
        s=Const.MARKER_SIZE,
    )
    ax.scatter(
        t_b_nn,
        b_c_nn,
        label="$b_c$",
        s=Const.MARKER_SIZE,
    )
    fig.show()
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused_s",
    )
    ax.plot(signal_generator.t, signal_generator.s, label=r"$s$")
    fig.show()

    """
        Training
    """
    elbo = fusion_model.iter_elbo
    fig, ax = plot_signals(
        [(np.arange(elbo.size), elbo, r"ELBO", False)],
        results_dir=results_dir,
        title="iter-elbo",
        legend="lower right",
        tight_layout=True,
    )
    fig.show()

    history = fusion_model.history
    if history:
        n_evals = len(history)
        history = [
            (t_out, mean, f"{i}/{n_evals}", False)
            for i, (mean, std) in enumerate(history)
        ]
        fig, ax = plot_signals(
            history,
            results_dir=results_dir,
            title="signals_fused_history",
            legend="lower right",
            x_ticker=1,
            tight_layout=True,
        )
        fig.show()
