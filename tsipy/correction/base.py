from enum import Enum, auto

from tsipy.correction.algorithms import correct_one, correct_both


class CorrectionMethod(Enum):
    CORRECT_ONE = auto()
    CORRECT_BOTH = auto()


def correct_degradation(
    t_m,
    a_m,
    e_a_m,
    b_m,
    e_b_m,
    model,
    method=CorrectionMethod.CORRECT_ONE,
    verbose=False,
    **kwargs
):
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
