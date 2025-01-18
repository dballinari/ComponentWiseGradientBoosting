import numpy as np

from cwgboost.boosting import CWGBoost
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Generate synthetic data with 10 features and 100 samples
    np.random.seed(42)
    x = np.random.uniform(-4, 4, (100, 10))
    y = np.sin(np.maximum(x[:, 0],0)) + np.random.normal(0, 0.2, len(x))
    
    # Fit CWGBoost
    model = CWGBoost(learning_rate=0.1, num_steps=1000, degree=3, n_knots=15, penalty_order=2, lambda_penalty=10000)
    model.fit(x, y)
    result = model.predict(x)
    
    plt.scatter(x[:, 0], y, color="gray", label="Data", alpha=0.6)
    plt.scatter(x[:, 0], result, color="red", label="CWGBoost fit")
    plt.legend()
    plt.show()
    
    