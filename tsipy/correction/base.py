from enum import Enum, auto

from .algorithms import correct_one, correct_both
from .models import load_model, DegradationModel


class CorrectionMethod(Enum):
    CORRECT_ONE = auto()
    CORRECT_BOTH = auto()


def correct_degradation(
    t_m,
    a_m,
    e_a_m,
    b_m,
    e_b_m,
    degradation_model=DegradationModel.SMR,
    method=CorrectionMethod.CORRECT_ONE,
    **kwargs
):
    model = load_model(degradation_model)
    model.initial_fit(a_m, b_m, e_a_m)
    model.convex = True if method == CorrectionMethod.CORRECT_ONE else False

    if method == CorrectionMethod.CORRECT_ONE:
        a_m_c, b_m_c, model, history = correct_one(
            t_m, a_m, e_a_m, b_m, e_b_m, model, **kwargs
        )
    elif method == CorrectionMethod.CORRECT_BOTH:
        a_m_c, b_m_c, model, history = correct_both(
            t_m, a_m, e_a_m, b_m, e_b_m, model, **kwargs
        )
    else:
        raise ValueError("Invalid correction method.")

    return a_m_c, b_m_c, model, history
