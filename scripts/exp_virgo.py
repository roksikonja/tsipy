import os

import gpflow as gpf
import numpy as np
import tensorflow as tf

import tsipy.correction
import tsipy.fusion
from tsipy.fusion.utils import (
    build_labels,
    build_output_labels,
    concatenate_labels,
)
from tsipy.utils import pprint, pprint_block
from utils.data import (
    create_results_dir,
    load_data,
    get_time_output,
    downsampling_indices_by_max_points,
)
from utils.visualizer import (
    plot_signals,
    plot_signals_and_confidence,
    plot_signals_history,
)

if __name__ == "__main__":
    results_dir = create_results_dir("../results", "exp_virgo")

    """
        Parameters
    """
    exposure_method = tsipy.correction.ExposureMethod.NUM_MEASUREMENTS
    correction_method = tsipy.correction.CorrectionMethod.CORRECT_ONE
    degradation_model = "explin"

    """
        Dataset
    """
    pprint_block("Virgo Dataset")
    np.random.seed(0)
    tf.random.set_seed(0)

    t_field = "t"
    e_field = "e"

    a_field = "a"
    b_field = "b"

    t_a_field, t_b_field = t_field + "_a", t_field + "_b"
    e_a_field, e_b_field = e_field + "_a", e_field + "_b"

    # Load data
    data = load_data("../data", "virgo_2020.h5")

    a = data[a_field].values
    b = data[b_field].values
    t = data[t_field].values

    print(data, "\n")
    pprint("- data", data.shape)
    pprint("- " + t_field, t.shape)
    pprint("- " + a_field, a.shape, np.sum(~np.isnan(a)))
    pprint("- " + b_field, b.shape, np.sum(~np.isnan(b)))

    # Compute exposure
    e_a = tsipy.correction.compute_exposure(a, method=exposure_method)
    e_b = tsipy.correction.compute_exposure(b, method=exposure_method)
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

    fig, _ = plot_signals(
        [
            (t_a_nn, a_nn, r"$a$", False),
            (t_b_nn, b_nn, r"$b$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        x_ticker=4,
        y_lim=[1357, 1369],
    )
    fig.show()

    """
        Degradation correction
    """
    pprint_block("Degradation Correction")
    degradation_model = tsipy.correction.load_model(degradation_model)
    degradation_model.initial_fit(a_m, b_m, e_a_m)

    a_m_c, b_m_c, degradation_model, history = tsipy.correction.correct_degradation(
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
        ],
        results_dir=results_dir,
        title="signals_corrected",
        legend="upper right",
        x_ticker=4,
    )
    fig.show()

    fig, _ = plot_signals(
        [
            (t_a_nn, d_a_c, r"$d(e_a(t))$", False),
            (t_b_nn, d_b_c, r"$d(e_b(t))$", False),
        ],
        results_dir=results_dir,
        title="degradation",
        legend="lower left",
        x_ticker=4,
    )
    fig.show()

    fig, _ = plot_signals_history(
        t_m,
        [
            [
                (signals[0], r"$a_{}$".format(i)),
                (signals[1], r"$b_{}$".format(i)),
            ]
            for i, signals in enumerate(history[:4])
        ],
        results_dir,
        title="correction-history",
        n_rows=2,
        n_cols=2,
        x_ticker=4,
    )
    fig.show()

    """
        Data fusion
    """
    pprint_block("Data Fusion")
    gpf.config.set_default_float(np.float64)
    np.random.seed(0)
    tf.random.set_seed(0)

    pprint("- t_a_nn", t_a_nn.shape)
    pprint("- t_b_nn", t_b_nn.shape)
    pprint("- a_c_nn", a_c_nn.shape)
    pprint("- b_c_nn", b_c_nn.shape)

    labels, t_labels = build_labels((t_a_nn, t_b_nn))
    s = np.hstack((a_c_nn, b_c_nn))
    t = np.hstack((t_a_nn, t_b_nn))
    t = concatenate_labels(t, t_labels, sort_axis=0)

    pprint("- labels", labels)
    pprint("- t_labels", t_labels.shape)
    pprint("- t", t.shape)
    pprint("- s", s.shape)

    t_out = get_time_output((t_a_nn, t_b_nn), n_out_per_unit=365 * 24)
    t_out_labels = build_output_labels(t_out)
    t_out = concatenate_labels(t_out, t_out_labels, sort_axis=0)

    pprint("- t_out_labels", t_out_labels.shape)
    pprint("- t_out", t_out.shape)

    """
        Kernel
    """
    # Signal kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])  # Kernel for time dimension

    # Noise kernel
    white_kernel = tsipy.fusion.kernels.MultiWhiteKernel(
        labels=labels, active_dims=[1]
    )  # Kernel for sensor dimension

    # Kernel composite
    kernel = matern_kernel + white_kernel

    """
        Gaussian Process Model
    """
    fusion_model = tsipy.fusion.models_gp.SVGPModel(
        kernel=kernel, num_inducing_pts=1000
    )

    # Train
    fusion_model.fit(t, s, max_iter=8000, verbose=True)

    # Predict
    s_out_mean, s_out_std = fusion_model(t_out)
    t_out = t_out[:, 0]

    pprint("- t_out", t_out.shape)
    pprint("- s_out_mean", s_out_mean.shape)
    pprint("- s_out_std", s_out_std.shape)

    fig, ax = plot_signals_and_confidence(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
        x_ticker=4,
        y_lim=[1362, 1369],
    )
    fig.show()
    indices_a = downsampling_indices_by_max_points(
        t_a_nn, max_points=2e4
    )  # Downsample signal a for plotting
    ax.scatter(
        t_a_nn[indices_a],
        a_c_nn[indices_a],
        label=r"$a$",
        s=3,
    )
    ax.scatter(
        t_b_nn,
        b_c_nn,
        label=r"$b$",
        s=3,
    )
    fig.show()
    fig.savefig(os.path.join(results_dir, "signals_fused_points"))

    """
        Training
    """
    elbo = fusion_model.iter_elbo
    fig, ax = plot_signals(
        [(np.arange(elbo.size), elbo, r"ELBO", False)],
        results_dir=results_dir,
        title="iter_elbo",
        legend="lower right",
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
            x_ticker=1,
        )
        fig.show()
