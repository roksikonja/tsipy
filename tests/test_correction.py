import numpy as np

import tsipy.correction
from tests.utils import check_array_approximate


def test_degradation_correct_one() -> None:
    """Tests convergence of iterative correction algorithm in the absence of noise.

    It tests correction method `correct_one` and all degradation models over multiple
    random seeds.
    """
    for random_seed in range(5):
        for model_name in ["exp", "explin", "mr", "smr"]:
            _run_degradation_correction(
                random_seed=random_seed,
                correction_method="correct_one",
                model_name=model_name,
            )


def test_degradation_correct_both() -> None:
    """Tests convergence of iterative correction algorithm in the absence of noise.

    It tests correction method `correct_both` and all degradation models over multiple
    random seeds.
    """
    for random_seed in range(5):
        for model_name in ["exp", "mr", "smr"]:
            _run_degradation_correction(
                random_seed=random_seed,
                correction_method="correct_both",
                model_name=model_name,
            )


def _run_degradation_correction(
    random_seed: int, correction_method: str, model_name: str, tolerance: float = 0.01
) -> None:
    """Runs degradation correction and checks convergence."""
    np.random.seed(random_seed)

    signal_generator = tsipy.correction.SignalGenerator(
        add_noise=False,
        downsampling_rates=(0.99, 0.2),
        random_seed=random_seed,
    )
    data = signal_generator.data

    a = data["a"].values
    b = data["b"].values

    # Compute exposure
    e_a = tsipy.correction.compute_exposure(a)
    e_b = tsipy.correction.compute_exposure(b)
    e_a /= signal_generator.length
    e_b /= signal_generator.length
    data["e_a"] = e_a
    data["e_b"] = e_b

    # Channel measurements
    _, a_nn, e_a_nn = signal_generator.get_signal_nn("a")
    _, b_nn, e_b_nn = signal_generator.get_signal_nn("b")

    # Mutual measurements
    data_m = data[["t", "a", "b", "e_a", "e_b"]].dropna()
    t_m = data_m["t"].values
    a_m, b_m = data_m["a"].values, data_m["b"].values
    e_a_m, e_b_m = data_m["e_a"].values, data_m["e_b"].values

    degradation_model = tsipy.correction.load_model(model_name)
    degradation_model.initial_fit(x_a=e_a_m, y_a=a_m, y_b=b_m)

    *_, degradation_model, _ = tsipy.correction.correct_degradation(
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

    a_ids_nn = signal_generator.get_indices_nn("a")
    b_ids_nn = signal_generator.get_indices_nn("b")

    # Check exposure computation
    check_array_approximate(signal_generator.get_exposure_nn("a"), e_a_nn)
    check_array_approximate(signal_generator.get_exposure_nn("b"), e_b_nn)

    # Check convergence
    check_array_approximate(signal_generator.y[a_ids_nn], a_c_nn, tolerance=tolerance)
    check_array_approximate(signal_generator.y[b_ids_nn], b_c_nn, tolerance=tolerance)
