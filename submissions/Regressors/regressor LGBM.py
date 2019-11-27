from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = LGBMRegressor(num_leaves=100,
                   learning_rate=0.01,
                   n_estimators=1000,
                   max_depth=-1)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
