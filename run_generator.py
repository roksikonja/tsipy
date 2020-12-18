import os

import gpflow as gpf
import numpy as np
import tensorflow as tf

import tsipy
from tsipy.correction import (
    ExposureMethod,
    compute_exposure,
    DegradationModel,
    CorrectionMethod,
    correct_degradation,
    SignalGenerator,
)
from tsipy.fusion import (
    FusionModel,
    MultiWhiteKernel,
    build_sensor_labels,
    build_output_labels,
    concatenate_labels,
)
from utils import create_results_dir, Constants as Const
from utils.visualizer import (
    pprint,
    plot_signals,
    plot_signals_mean_std_precompute,
)

if __name__ == "__main__":
    results_dir = create_results_dir(Const.RESULTS_DIR, "virgo")

    """
        Parameters
    """
    exposure_method = ExposureMethod.NUM_MEASUREMENTS
    correction_method = CorrectionMethod.CORRECT_ONE
    degradation_model = DegradationModel.SMR
    fusion_model = FusionModel.SVGP

    """
        Dataset
    """
    np.random.seed(Const.RANDOM_SEED)

    t_field = "t"
    e_field = "e"

    t_a_field, t_b_field = t_field + "_a", t_field + "_b"
    a_field, b_field = "a", "b"
    e_a_field, e_b_field = e_field + "_a", e_field + "_b"

    #
    signal_generator = SignalGenerator()
    data = signal_generator.data

    a = data[a_field].values
    b = data[b_field].values
    t = data[t_field].values

    pprint(t_field, t.shape)
    pprint(a_field, a.shape, np.sum(~np.isnan(a)))
    pprint(b_field, b.shape, np.sum(~np.isnan(b)))

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

    pprint(t_a_field, t_a_nn.min(), t_a_nn.max())
    pprint(t_b_field, t_b_nn.min(), t_b_nn.max())

    # Mutual measurements
    data_m = data[[t_field, a_field, b_field, e_a_field, e_b_field]].dropna()
    t_m = data_m[t_field].values
    a_m, b_m = data_m[a_field].values, data_m[b_field].values
    e_a_m, e_b_m = data_m[e_a_field].values, data_m[e_b_field].values

    pprint(a_field, a_m.shape, np.sum(~np.isnan(a_m)))
    pprint(b_field, b_m.shape, np.sum(~np.isnan(b_m)))
    pprint(e_a_field, e_a_m.shape)
    pprint(e_b_field, e_b_m.shape, "\n")

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
    )

    d_a_c = degradation_model(e_a_nn)
    d_b_c = degradation_model(e_b_nn)
    a_c_nn = np.divide(a_nn, d_a_c)
    b_c_nn = np.divide(b_nn, d_b_c)

    fig, _ = plot_signals(
        [
            (t_m, a_m, r"$a$", False),
            (t_m, b_m, r"$b$", False),
            (signal_generator.t, signal_generator.s, r"$s$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        y_label=Const.TSI_UNIT,
    )
    fig.show()

    fig, _ = plot_signals(
        [
            (t_m, a_m_c, r"$a_c$", False),
            (t_m, b_m_c, r"$b_c$", False),
            (signal_generator.t, signal_generator.s, r"$s$", False),
        ],
        results_dir=results_dir,
        title="signals_corrected",
        legend="upper right",
        y_label=Const.TSI_UNIT,
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
        y_label=Const.DEGRADATION_UNIT,
    )
    fig.show()

    """
            Data fusion
        """
    gpf.config.set_default_float(np.float64)
    np.random.seed(Const.RANDOM_SEED)
    tf.random.set_seed(Const.RANDOM_SEED)

    pprint("t_a_nn", t_a_nn.shape)
    pprint("t_b_nn", t_b_nn.shape)
    pprint("a_c_nn", a_c_nn.shape)
    pprint("b_c_nn", b_c_nn.shape)

    labels, t_labels = build_sensor_labels((t_a_nn, t_b_nn))
    s = np.hstack((a_c_nn, b_c_nn))
    t = np.hstack((t_a_nn, t_b_nn))
    t = concatenate_labels(t, t_labels)

    pprint("labels", labels)
    pprint("t_labels", t_labels.shape)
    pprint("t", t.shape)
    pprint("s", s.shape)

    t_out = signal_generator.t
    t_out_labels = build_output_labels(t_out)
    t_out = concatenate_labels(t_out, t_out_labels)

    pprint("x_out_labels", t_out_labels.shape)
    pprint("t_out", t_out.shape)

    # Kernel
    matern_kernel = gpf.kernels.Matern12(active_dims=[0])  # Kernel for time dimension

    cond = True
    if cond:
        white_kernel = MultiWhiteKernel(
            labels=labels, active_dims=[1]
        )  # Kernel for sensor dimension
    else:
        white_kernel = gpf.kernels.White(active_dims=[1])

    kernel = matern_kernel + white_kernel

    fusion_model = tsipy.fusion.load_model(
        fusion_model, kernel=kernel, num_inducing_pts=1000
    )
    fusion_model.fit(t, s, max_iter=2000, verbose=True)
    print(fusion_model)

    # Predict
    s_out_mean, s_out_std = fusion_model(t_out)
    t_out = t_out[:, 0]

    pprint("t_out", t_out.shape)
    pprint("s_out_mean", s_out_mean.shape)
    pprint("s_out_std", s_out_std.shape)

    fig, ax = plot_signals_mean_std_precompute(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused",
        legend="upper left",
        y_label=Const.TSI_UNIT,
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

    fig, ax = plot_signals_mean_std_precompute(
        [(t_out, s_out_mean, s_out_std, "SVGP")],
        results_dir=results_dir,
        title="signals_fused_s",
        legend="upper left",
        y_label=Const.TSI_UNIT,
    )
    ax.plot(signal_generator.t, signal_generator.s, label=r"$s$")
    fig.show()
