
import numpy as np
from scipy.linalg import solve
from scipy.interpolate import BSpline

def create_b_spline_basis(x, knots, degree):
    """Creates a B-spline basis matrix."""
    n_bases = len(knots) - (degree + 1)
    basis = np.zeros((len(x), n_bases))

    for i in range(n_bases):
        coeff = np.zeros(n_bases)
        coeff[i] = 1
        spline = BSpline(knots, coeff, degree)
        basis[:, i] = spline(x)

    return basis

def difference_matrix(n_bases, order=2):
    """Creates a difference penalty matrix of a given order."""
    D = np.eye(n_bases)
    for i in range(order):
        D = np.diff(D, axis=0)
    return D

class PSpline:
    def __init__(self, degree=3, n_knots=20, penalty_order=2, lambda_penalty=1):
        self.degree = degree
        self.n_knots = n_knots
        self.penalty_order = penalty_order
        self.lambda_penalty = lambda_penalty

    def fit(self, x, y):
        x_min, x_max = np.min(x), np.max(x)
        knots = np.linspace(x_min, x_max, self.n_knots)
        knots = np.concatenate((np.full(self.degree, x_min), knots, np.full(self.degree, x_max)))

        B = create_b_spline_basis(x, knots, self.degree)
        D = difference_matrix(B.shape[1], order=self.penalty_order)
        P = self.lambda_penalty * (D.T @ D)

        BtB = B.T @ B
        Bty = B.T @ y
        self.coeff_ = solve(BtB + P, Bty)
        self.knots_ = knots

    def predict(self, x):
        return create_b_spline_basis(x, self.knots_, self.degree) @ self.coeff_
