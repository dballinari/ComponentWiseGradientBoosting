
import numpy as np
from scipy.linalg import solve, sqrtm
from scipy.interpolate import BSpline
from scipy.optimize import bisect


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
    def __init__(self, penalty_order=2, df=4, lambda_penalty=None):
        self.df = df
        self.penalty_order = penalty_order
        self.lambda_penalty = lambda_penalty

    def fit(self, B, y):
        D = difference_matrix(B.shape[1], order=self.penalty_order)
        DtD = D.T @ D
        
        BtB = B.T @ B
        Bty = B.T @ y
        
         # find penalty to match degrees of freedom
        if self.lambda_penalty is None:
            Qb =  sqrtm(np.linalg.inv(BtB))
            L = Qb @ DtD @ Qb
            self.lambda_penalty = bisect(lambda x: np.linalg.trace( np.linalg.inv(np.eye(L.shape[0])+x * L) ) - self.df, 0, 1e11)
        
        P = self.lambda_penalty * DtD
        self.coeff_ = solve(BtB + P, Bty)

    def predict(self, B):
        return B @ self.coeff_
