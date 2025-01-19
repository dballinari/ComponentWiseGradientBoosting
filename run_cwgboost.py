import numpy as np

from cwgboost.boosting import CWGBoost
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Generate synthetic data with 10 features and 100 samples
    np.random.seed(42)
    x = np.random.uniform(-4, 4, (100, 10))
    y = np.sin(np.minimum(x[:, 1],0)) * np.sin(np.maximum(x[:, 0],0)) + np.random.normal(0, 0.2, len(x))
    x_val = np.random.uniform(-4, 4, (100, 10))
    y_val = np.sin(np.minimum(x_val[:, 1],0)) * np.sin(np.maximum(x_val[:, 0],0)) + np.random.normal(0, 0.2, len(x))
    
    # Fit CWGBoost
    model = CWGBoost(learning_rate=0.1, num_steps=5000, degree=3, n_knots=20, penalty_order=2, lambda_penalty=10)
    model.fit(x, y, verbose=True, x_val=x_val, y_val=y_val, early_stopping_rounds=5)
    
    x_test = np.random.uniform(-4, 4, (100, 10))
    y_test = np.sin(np.minimum(x_test[:, 1],0)) * np.sin(np.maximum(x_test[:, 0],0)) + np.random.normal(0, 0.2, len(x))
    result = model.predict(x_test)
    print(((result - y_test)**2).mean())
    
    