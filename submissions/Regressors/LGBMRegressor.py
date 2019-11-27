from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = LGBMRegressor(num_leaves=30,
                    boosting_type='gbdt',
                    objective='regression',
                    learning_rate=0.1,
                    max_depth=-1,
                    n_estimators=400,
                    bagging_fraction=0.52,
                    feature_fraction=0.63,
                    max_bin=255)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
