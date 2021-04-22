def test_dependencies() -> None:
    """Tests if all dependencies are installed and there are no conflicts."""
    # flake8: noqa: F401

    import os
    import warnings

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import gpflow
    import matplotlib
    import numpy
    import pandas
    import qpsolvers
    import scipy
    import sklearn
    import tables
    import tensorflow

    import tsipy


def test_qpsolvers() -> None:
    """Tests cvxpy and numpy conflicts."""
    import numpy as np
    from qpsolvers import solve_qp

    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    P = np.dot(M.T, M)  # this is a positive definite matrix
    q = np.dot(np.array([3.0, 2.0, 3.0]), M).reshape((3,))
    G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
    h = np.array([3.0, 2.0, -2.0]).reshape((3,))
    A = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0])

    solve_qp(P, q, G, h, A, b)
