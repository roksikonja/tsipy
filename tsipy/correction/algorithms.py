import numpy as np

from ..utils import pprint


def correct_one(
    t_m,
    a_m,
    e_a_m,
    b_m,
    e_b_m,
    model,
    verbose=False,
    eps=1e-6,
    max_iter=100,
):
    ratio_m = np.divide(a_m, b_m + 1e-9)
    correction_triplet = (a_m, b_m, ratio_m)
    history = [correction_triplet]

    a_m_c, b_m_c = a_m, b_m
    i = 0
    converged = False
    while i < max_iter and not converged:
        previous_correction_triplet = correction_triplet

        model.fit(ratio_m, e_a_m)
        d_a_c, d_b_c = model(e_a_m), model(e_b_m)

        a_m_c = np.divide(a_m, d_a_c + 1e-9)
        b_m_c = np.divide(b_m, d_b_c + 1e-9)
        ratio_m = np.divide(a_m, b_m_c + 1e-9)

        correction_triplet = (a_m_c, b_m_c, ratio_m)
        history.append(correction_triplet)

        # Check convergence
        previous_a_m_c = previous_correction_triplet[0]
        previous_b_m_c = previous_correction_triplet[1]

        delta_norm_a = np.linalg.norm(a_m_c - previous_a_m_c) / np.linalg.norm(
            previous_a_m_c
        )
        delta_norm_b = np.linalg.norm(b_m_c - previous_b_m_c) / np.linalg.norm(
            previous_b_m_c
        )

        converged = delta_norm_a + delta_norm_b < eps
        i = i + 1

    if verbose:
        pprint(f"    - Corrected in {i} iterations")

    return a_m_c, b_m_c, model, history


def correct_both(
    t_m,
    a_m,
    e_a_m,
    b_m,
    e_b_m,
    model,
    verbose=False,
    eps=1e-6,
    max_iter=100,
):
    ratio_m = np.divide(a_m, b_m + 1e-9)
    correction_triplet = (a_m, b_m, ratio_m)
    history = [correction_triplet]

    a_m_c, b_m_c = a_m, b_m
    i = 0
    converged = False
    while i < max_iter and not converged:
        previous_correction_triplet = correction_triplet

        model.fit(ratio_m, e_a_m)
        d_a_c, d_b_c = model(e_a_m), model(e_b_m)

        a_m_c = np.divide(a_m_c, d_a_c + 1e-9)
        b_m_c = np.divide(b_m_c, d_b_c + 1e-9)
        ratio_m = np.divide(a_m_c, b_m_c + 1e-9)

        correction_triplet = (a_m_c, b_m_c, ratio_m)
        history.append(correction_triplet)

        # Check convergence
        previous_a_m_c = previous_correction_triplet[0]
        previous_b_m_c = previous_correction_triplet[1]

        delta_norm_a = np.linalg.norm(a_m_c - previous_a_m_c) / np.linalg.norm(
            previous_a_m_c
        )
        delta_norm_b = np.linalg.norm(b_m_c - previous_b_m_c) / np.linalg.norm(
            previous_b_m_c
        )

        converged = delta_norm_a + delta_norm_b < eps
        i = i + 1

    if verbose:
        print(f"corrected in {i} iterations")

    # Re-fit
    ratio_m = np.divide(a_m, b_m_c + 1e-9)
    model.fit(ratio_m, e_a_m)
    d_a_c, d_b_c = model(e_a_m), model(e_b_m)

    a_m_c = np.divide(a_m, d_a_c + 1e-9)
    b_m_c = np.divide(b_m, d_b_c + 1e-9)

    return a_m_c, b_m_c, model, history
