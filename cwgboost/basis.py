from scipy.interpolate import BSpline
import numpy as np


def create_b_spline_basis(x, knots, degree):
    """Creates a B-spline basis matrix."""
    n_obs = x.shape[0]
    n_features = x.shape[1]
    n_bases = len(knots) - (degree + 1)
    basis = np.zeros((n_obs, n_bases, n_features))
    for feature in range(n_features):
        for i in range(n_bases):
            coeff = np.zeros(n_bases)
            coeff[i] = 1
            spline = BSpline(knots[:,feature], coeff, degree)
            basis[:, i, feature] = spline(x[:, feature])
    return basis