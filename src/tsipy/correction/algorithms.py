"""
This module implements algorithms that perform degradation correction.
"""

from typing import List, Tuple
from collections import namedtuple

import numpy as np

from .models import DegradationModel
from ..utils import pprint

History = namedtuple("History", ["iteration", "a", "b", "ratio"])


def correct_degradation(
    t_m: np.ndarray,
    a_m: np.ndarray,
    e_a_m: np.ndarray,
    b_m: np.ndarray,
    e_b_m: np.ndarray,
    model: DegradationModel,
    method: str = "correct_one",
    verbose: bool = False,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, DegradationModel, List[History]]:
    """Selects and executes a correction algorithm.

    This is a wrapper function for :func:`correct_one` and :func:`correct_both`.
    """
    # pylint: disable=R0913
    if method == "correct_one":
        a_m_c, b_m_c, model, history = correct_one(
            t_m=t_m,
            a_m=a_m,
            e_a_m=e_a_m,
            b_m=b_m,
            e_b_m=e_b_m,
            model=model,
            verbose=verbose,
            eps=eps,
            max_iter=max_iter,
        )
    elif method == "correct_both":
        a_m_c, b_m_c, model, history = correct_both(
            t_m=t_m,
            a_m=a_m,
            e_a_m=e_a_m,
            b_m=b_m,
            e_b_m=e_b_m,
            model=model,
            verbose=verbose,
            eps=eps,
            max_iter=max_iter,
        )
    else:
        raise ValueError("Invalid correction method.")

    return a_m_c, b_m_c, model, history


def correct_one(
    t_m: np.ndarray,
    a_m: np.ndarray,
    e_a_m: np.ndarray,
    b_m: np.ndarray,
    e_b_m: np.ndarray,
    model: DegradationModel,
    verbose: bool = False,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, DegradationModel, List[History]]:
    """Executes degradation correction algorithm ``CorrectOne``.

    The algorithm is described in
    `Kolar, Šikonja and Treven, 2020 <https://arxiv.org/abs/2009.03091>`_.
    It is shown that corrected signals converge to the ground truth in the absence of
    measurement noise.

    Returns:
        Corrected signals ``a`` and ``b``, degradation model ``d_c(.)`` and correction
        history.
    """
    # pylint: disable=R0913, R0914
    del t_m

    ratio_m = np.divide(a_m, b_m + 1e-9)
    correction_triplet = History(0, a_m, b_m, ratio_m)
    history = [correction_triplet]

    a_m_c, b_m_c = a_m, b_m
    iteration = 1
    for iteration in range(1, max_iter + 1):
        previous_correction_triplet = correction_triplet

        model.fit(ratio_m, e_a_m)
        d_a_c, d_b_c = model(e_a_m), model(e_b_m)

        a_m_c = np.divide(a_m, d_a_c + 1e-9)
        b_m_c = np.divide(b_m, d_b_c + 1e-9)
        ratio_m = np.divide(a_m, b_m_c + 1e-9)

        correction_triplet = (a_m_c, b_m_c, ratio_m)
        history.append(correction_triplet)

        correction_triplet = History(iteration, a_m_c, b_m_c, ratio_m)
        history.append(correction_triplet)

        converged = _check_convergence(
            a=a_m_c,
            b=b_m_c,
            ref_a=previous_correction_triplet.a,
            ref_b=previous_correction_triplet.b,
            eps=eps,
        )
        if converged:
            break

    if verbose:
        pprint(f"- Corrected in {iteration} iterations.", level=1)

    return a_m_c, b_m_c, model, history


def correct_both(
    t_m: np.ndarray,
    a_m: np.ndarray,
    e_a_m: np.ndarray,
    b_m: np.ndarray,
    e_b_m: np.ndarray,
    model: DegradationModel,
    verbose: bool = False,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, DegradationModel, List[History]]:
    """Executes degradation correction algorithm ``CorrectBoth``.

    The algorithm is described in
    `Kolar, Šikonja and Treven, 2020 <https://arxiv.org/abs/2009.03091>`_.
    It is shown that corrected signals converge to the ground truth in the absence of
    measurement noise.

    Returns:
        Corrected signals ``a`` and ``b``, degradation model ``d_c(.)`` and correction
        history.
    """
    # pylint: disable=R0913, R0914
    del t_m

    ratio_m = np.divide(a_m, b_m + 1e-9)
    correction_triplet = History(0, a_m, b_m, ratio_m)
    history = [correction_triplet]

    a_m_c, b_m_c = a_m, b_m
    iteration = 1
    for iteration in range(1, max_iter + 1):
        previous_correction_triplet = correction_triplet

        model.fit(ratio_m, e_a_m)
        d_a_c, d_b_c = model(e_a_m), model(e_b_m)

        a_m_c = np.divide(a_m_c, d_a_c + 1e-9)
        b_m_c = np.divide(b_m_c, d_b_c + 1e-9)
        ratio_m = np.divide(a_m_c, b_m_c + 1e-9)

        correction_triplet = History(iteration, a_m_c, b_m_c, ratio_m)
        history.append(correction_triplet)

        converged = _check_convergence(
            a=a_m_c,
            b=b_m_c,
            ref_a=previous_correction_triplet.a,
            ref_b=previous_correction_triplet.b,
            eps=eps,
        )
        if converged:
            break

    if verbose:
        pprint(f"- Corrected in {iteration} iterations.", level=1)

    # Re-fit
    # In CorrectBoth, model converges to a constant function of 1
    ratio_m = np.divide(a_m, b_m_c + 1e-9)
    model.fit(ratio_m, e_a_m)
    d_a_c, d_b_c = model(e_a_m), model(e_b_m)

    a_m_c = np.divide(a_m, d_a_c + 1e-9)
    b_m_c = np.divide(b_m, d_b_c + 1e-9)

    return a_m_c, b_m_c, model, history


def _check_convergence(
    a: np.ndarray,
    b: np.ndarray,
    ref_a: np.ndarray,
    ref_b: np.ndarray,
    eps: float = 1e-6,
) -> bool:
    """Computes relative difference between consecutive steps and checks convergence."""
    # pylint: disable=C0103
    delta_norm_a = np.linalg.norm(a - ref_a) / np.linalg.norm(ref_a)
    delta_norm_b = np.linalg.norm(b - ref_b) / np.linalg.norm(ref_b)

    return delta_norm_a + delta_norm_b < eps
