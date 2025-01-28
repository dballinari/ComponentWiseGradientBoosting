import numpy as np
from .psplines import PSpline
from .basis import create_b_spline_basis
from enum import Enum
from sklearn.tree import DecisionTreeRegressor

class BaseLearner(Enum):
    PSPLINE = 1
    TREESTUMP = 2
    ALL = 3


class CWGBoost:
    def __init__(self, learning_rate: float=0.1, num_steps: int=100, base_learner: BaseLearner=BaseLearner.PSPLINE, degree: int=3, n_knots: int=20, penalty_order: int=2, df: int=4):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.degree = degree
        self.n_knots = n_knots
        self.penalty_order = penalty_order
        self.df = df
        self.base_learner = base_learner

    def fit(self, x, y, x_val=None, y_val=None, early_stopping_rounds=None, verbose=False):
        self._fit_knots(x)
        if self.base_learner == BaseLearner.PSPLINE:
            basis = create_b_spline_basis(x, self.knots, self.degree)
            if x_val is not None:
                basis_val = create_b_spline_basis(x_val, self.knots, self.degree)
        else:
            basis = x.reshape(-1, 1, x.shape[1])
            if x_val is not None:
                basis_val = x_val.reshape(-1, 1, x_val.shape[1])
            
        # lambda penalty is determined as a function of the degrees of freedom, we or each feature we
        # store this penalty term such that it has not to be recomputed at each iteration
        lambda_penalties = [None] * x.shape[1]
        
        self.mean_ = np.mean(y)
        
        f = np.full(len(y), self.mean_)
        self.models = []
        
        # Initialize validation prediction
        f_val = np.full(len(y_val), self.mean_)
        validation_loss = np.zeros(self.num_steps)
        
        for step_i in range(self.num_steps):
            # only implements the squared loss where the gradient is the residual
            residuals = y - f

            best_base_learner = None
            best_loss = np.mean(residuals ** 2)
            best_pred_base_learner = None
            best_feature = None
            
            if verbose and (step_i + 1) % 10 == 0:
                print(f"Step {step_i + 1}/{self.num_steps}")
            
            # Iterate over each feature
            for i in range(x.shape[1]):
                if self.base_learner == BaseLearner.TREESTUMP:
                    base_learner = DecisionTreeRegressor(max_depth=1)
                    base_learner = base_learner.fit(basis[:, :, i], residuals)
                elif self.base_learner == BaseLearner.PSPLINE:
                    # Fit a P-Spline to the residuals: at first iteration, the lambda penalty is None and will be computed based on the degrees of freedom
                    base_learner = PSpline(penalty_order=self.penalty_order, df=self.df, lambda_penalty=lambda_penalties[i])
                    base_learner.fit(basis[:, :, i], residuals)
                    # store the lambda penalty for the next iteration
                    lambda_penalties[i] = base_learner.lambda_penalty
                else:
                    raise ValueError("Invalid base learner")
                pred_base_learner = base_learner.predict(basis[:, :, i])
                # Check if base learner is better than the current best
                loss_base_learner = np.mean((y - (f + self.learning_rate * pred_base_learner))**2)
                if loss_base_learner < best_loss:
                    best_base_learner = base_learner
                    best_loss = loss_base_learner
                    best_pred_base_learner = pred_base_learner
                    best_feature = i
            if best_base_learner is None:
                break
            self.models.append({'feature': best_feature, 'model': best_base_learner})
            f += self.learning_rate * best_pred_base_learner
            
            # Check early stopping
            if early_stopping_rounds is not None:
                # update validation prediction
                f_val += self.learning_rate * best_base_learner.predict(basis_val[:, :, best_feature])
                validation_loss[step_i] = np.mean((y_val - f_val)**2)
                if step_i > early_stopping_rounds:
                    # if the last early_stopping_rounds validation losses are higher than the previous ones, stop
                    if np.all(validation_loss[step_i - early_stopping_rounds:step_i] > validation_loss[step_i - early_stopping_rounds - 1]):
                        # remove the last models that did not improve the validation loss
                        self.models = self.models[:(step_i - early_stopping_rounds)]
                        break
    
                    
    def _fit_knots(self, x):
        x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
        knots = np.linspace(x_min, x_max, self.n_knots)
        self.knots = np.concatenate((np.full((self.degree, x.shape[1]), x_min), knots, np.full((self.degree, x.shape[1]), x_max)))

    def predict(self, x):
        if self.base_learner == BaseLearner.PSPLINE:
            basis = create_b_spline_basis(x, self.knots, self.degree)
        else:
            basis = x.reshape(-1, 1, x.shape[1])
        return self._predict_from_basis(basis)
    
    def _predict_from_basis(self, basis):
        f =  np.full(basis.shape[0], self.mean_)
        for model in self.models:
            f += self.learning_rate * model['model'].predict(basis[:, :, model['feature']])
        return f
    
    def feature_importance(self, x, y):
       
        if self.base_learner == BaseLearner.PSPLINE:
            basis = create_b_spline_basis(x, self.knots, self.degree)
        else:
            basis = x.reshape(-1, 1, x.shape[1])
        baseline_mse = np.mean((y - self._predict_from_basis(basis))**2)
          
        importance = {}
        for feature in range(x.shape[1]):
            f =  np.full(x.shape[0], self.mean_)
            for model in self.models:
                if model['feature'] != feature:
                    f += self.learning_rate * model['model'].predict(basis[:, :, model['feature']])
            importance[feature] =  (np.mean((y - f)**2) - baseline_mse)/baseline_mse
            
        return importance
