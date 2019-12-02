from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = LGBMRegressor(num_leaves=40,
                    boosting_type='gbdt',
                    objective='regression',
                    learning_rate=0.1,
                    max_depth=-1,
                    n_estimators=1800,
                    max_bin=255, silent=True,
                    reg_alpha=.001,
                    reg_lambda=.01)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
