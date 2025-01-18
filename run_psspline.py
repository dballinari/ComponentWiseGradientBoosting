import matplotlib.pyplot as plt
import numpy as np
from cwgboost.psplines import PSpline


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))

    # Fit P-spline
    model = PSpline(degree=3, n_knots=15, penalty_order=2, lambda_penalty=10)
    model.fit(x, y)
    result = model.predict(x)

    plt.scatter(x, y, color="gray", label="Data", alpha=0.6)
    plt.plot(x, result, color="red", label="P-spline fit")
    plt.legend()
    plt.show()
