import numpy as np
from sklearn.linear_model import LinearRegression
from cwgboost.boosting import CWGBoost, BaseLearner

def regime_switching_ar1(n: int):
    brun_in = 1000
    np.random.seed(42)
    x = np.random.normal(0, 0.5, n+brun_in)
    u = np.random.normal(0, 1, n+brun_in)
    y = np.zeros(n+brun_in)
    for i in range(1, n+brun_in):
        y[i] = (-0.5* (x[i]+y[i-1]>0) + 0.9 * (x[i]+y[i-i]<=0)) * y[i-1] + u[i]
    return y[-n:]


def prepare_date_for_regression(y):
    # regressors as the first 10 lags of y
    X = np.zeros((len(y)-10, 10))
    for i in range(10):
        X[:,i] = y[10-i-1:-(i+1)]
    y_regressors = y[10:]
    return X, y_regressors


y = regime_switching_ar1(6000)
x_train, y_train = prepare_date_for_regression(y[:4000])
print(x_train.shape, y_train.shape)

x_valid, y_valid = prepare_date_for_regression(y[4010:5000])
x_test, y_test = prepare_date_for_regression(y[5010:6000])


model = CWGBoost(learning_rate=0.1, num_steps=1000, base_learner=BaseLearner.PSPLINE, degree=2, n_knots=20, penalty_order=2, df=4)
model.fit(x_train, y_train, verbose=True, x_val=x_valid, y_val=y_valid, early_stopping_rounds=20)
print(model.feature_importance(x_train, y_train))

result = model.predict(x_test)
print(((result - y_test)**2).mean()/np.var(y_test))


model = CWGBoost(learning_rate=0.1, num_steps=1000, base_learner=BaseLearner.TREESTUMP)
model.fit(x_train, y_train, verbose=True, x_val=x_valid, y_val=y_valid, early_stopping_rounds=20)
print(model.feature_importance(x_train, y_train))

result = model.predict(x_test)
print(((result - y_test)**2).mean()/np.var(y_test))

# compare to AR(1) model
model = LinearRegression()
model.fit(x_train[:,[0]], y_train)
result = model.predict(x_test[:,[0]])
print(((result - y_test)**2).mean()/np.var(y_test))

