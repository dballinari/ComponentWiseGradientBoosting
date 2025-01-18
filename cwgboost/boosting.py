import numpy as np
from .psplines import PSpline


class CWGBoost:
    def __init__(self, learning_rate=0.1, num_steps=100, degree=3, n_knots=20, penalty_order=2, lambda_penalty=1):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.degree = degree
        self.n_knots = n_knots
        self.penalty_order = penalty_order
        self.lambda_penalty = lambda_penalty

    def fit(self, x, y, x_val=None, y_val=None, early_stopping_rounds=None, verbose=False):
        self.models = []
        self.mean_ = np.mean(y)
        
        f = np.full(len(y), self.mean_)
        
        validation_loss = np.zeros(self.num_steps)
        
        for step_i in range(self.num_steps):
            residuals = y - f

            best_base_learner = None
            best_loss = np.mean(residuals ** 2)
            best_pred_base_learner = None
            best_feature = None
            
            if verbose and (step_i + 1) % 10 == 0:
                print(f"Step {step_i + 1}/{self.num_steps}")
            
            # Iterate over each feature
            for i in range(x.shape[1]):
                # Fit a regression tree to the residuals
                base_learner = PSpline(degree=self.degree, n_knots=self.n_knots, penalty_order=self.penalty_order, lambda_penalty=self.lambda_penalty)
                base_learner.fit(x[:, i], residuals)
                pred_base_learner = base_learner.predict(x[:, i])
                # Check if base learner is better than the current best
                loss_base_learner = np.mean((y - (f + self.learning_rate * pred_base_learner))**2)
                if loss_base_learner < best_loss:
                    best_base_learner = base_learner
                    best_loss = loss_base_learner
                    best_pred_base_learner = pred_base_learner
                    best_feature = i
            
            self.models.append({'feature': best_feature, 'model': best_base_learner})
            f += self.learning_rate * best_pred_base_learner
            # Check early stopping
            if early_stopping_rounds is not None:
                validation_loss[step_i] = np.mean((y_val - self.predict(x_val))**2)
                if step_i > early_stopping_rounds:
                    # if the last early_stopping_rounds validation losses are higher than the previous ones, stop
                    if np.all(validation_loss[step_i - early_stopping_rounds:] > validation_loss[step_i - early_stopping_rounds - 1]):
                        break

    def predict(self, x):
        f =  np.full(x.shape[0], self.mean_)
        for model in self.models:
            f += self.learning_rate * model['model'].predict(x[:, model['feature']])
        return f
    
    def feature_importance(self, x, y):
        baseline_mse = np.mean((y - self.predict(x))**2)
        
        importance = {}
        for feature in range(x.shape[1]):
            f =  np.full(x.shape[0], self.mean_)
            for model in self.models:
                if model['feature'] != feature:
                    f += self.learning_rate * model['model'].predict(x[:, model['feature']])
            importance[feature] =  (np.mean((y - f)**2) - baseline_mse)/baseline_mse
            
        return importance
