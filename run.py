import numpy as np
import tensorflow as tf

from tsipy.correction import (
    ExposureMethod,
    compute_exposure,
    DegradationModel,
    CorrectionMethod,
    correct_degradation,
)
from tsipy.fusion import FusionModel, fuse_signals
from utils import create_results_dir
from utils.constants import Constants as Const
from utils.data import load_data, time_output
from utils.visualizer import pprint, plot_signals

if __name__ == "__main__":
    np.random.seed(Const.RANDOM_SEED)
    tf.random.set_seed(Const.RANDOM_SEED)

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
    t_field = "t"
    e_field = "e"

    t_a_field, t_b_field = t_field + "_a", t_field + "_b"
    a_field, b_field = "a", "b"
    e_a_field, e_b_field = e_field + "_a", e_field + "_b"

    # Load data
    data = load_data(Const.DATA_DIR, "virgo_level1_2020.h5")
    data = data.rename(
        columns={
            "TIME": t_field,
            "PMO6V-A": a_field,
            "PMO6V-B": b_field,
        }
    )

    a = data[a_field].values
    b = data[b_field].values
    t = data[t_field].values

    pprint(t_field, t.shape)
    pprint(a_field, a.shape, np.sum(~np.isnan(a)))
    pprint(b_field, b.shape, np.sum(~np.isnan(b)), "\n")

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
    a_m_c, b_m_c, degradation_model, history = correct_degradation(
        t_m,
        a_m,
        e_a_m,
        b_m,
        e_b_m,
        degradation_model=degradation_model,
        method=correction_method,
    )

    d_a_c = degradation_model(e_a_nn)
    d_b_c = degradation_model(e_b_nn)
    a_c_nn = np.divide(a_nn, d_a_c)
    b_c_nn = np.divide(b_nn, d_b_c)

    plot_signals(
        [
            (t_m, a_m, r"$a$", False),
            (t_m, b_m, r"$b$", False),
        ],
        results_dir=results_dir,
        title="signals",
        legend="upper right",
        x_ticker=Const.X_TICKER,
        x_label=Const.YEAR_UNIT,
        y_label=Const.TSI_UNIT,
    )

    plot_signals(
        [
            (t_m, a_m_c, r"$a_c$", False),
            (t_m, b_m_c, r"$b_c$", False),
        ],
        results_dir=results_dir,
        title="signals_corrected",
        legend="upper right",
        x_ticker=Const.X_TICKER,
        x_label=Const.YEAR_UNIT,
        y_label=Const.TSI_UNIT,
    )

    plot_signals(
        [
            (t_a_nn, d_a_c, r"$d(e_a(t))$", False),
            (t_b_nn, d_b_c, r"$d(e_b(t))$", False),
        ],
        results_dir=results_dir,
        title="degradation",
        legend="upper right",
        x_ticker=Const.X_TICKER,
        x_label=Const.YEAR_UNIT,
        y_label=Const.DEGRADATION_UNIT,
    )

    """
        Data fusion
        
        np.random.seed(Const.RANDOM_SEED)
        tf.random.set_seed(Const.RANDOM_SEED)
    
        t_out = time_output(t_a_nn, t_b_nn, n_out_per_unit=24)
        pprint("t_out", t_out)
    
        out_mean, out_std, fusion_model = fuse_signals(
            t_a_nn, t_b_nn, a_c_nn, b_c_nn, t_out, fusion_model=fusion_model
        )
    """
