from enum import Enum, auto

from .algorithms import correct_one, correct_both


class CorrectionMethod(Enum):
    CORRECT_ONE = auto()
    CORRECT_BOTH = auto()


def load_correction_method(method):
    if isinstance(method, CorrectionMethod):
        return method
    elif isinstance(method, str):
        if method == "correct_one":
            return CorrectionMethod.CORRECT_ONE
        elif method == "correct_both":
            return CorrectionMethod.CORRECT_BOTH

    raise ValueError("Invalid correction method.")


def correct_degradation(
    t_m, a_m, e_a_m, b_m, e_b_m, model, method="correct_one", verbose=False, **kwargs
):
    method = load_correction_method(method)

    model.convex = True if method == CorrectionMethod.CORRECT_ONE else False

    if method == CorrectionMethod.CORRECT_ONE:
        a_m_c, b_m_c, model, history = correct_one(
            t_m, a_m, e_a_m, b_m, e_b_m, model, verbose, **kwargs
        )
    elif method == CorrectionMethod.CORRECT_BOTH:
        a_m_c, b_m_c, model, history = correct_both(
            t_m, a_m, e_a_m, b_m, e_b_m, model, verbose, **kwargs
        )
    else:
        raise ValueError("Invalid correction method.")

    return a_m_c, b_m_c, model, history
