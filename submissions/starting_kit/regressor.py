from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.regXGB = XGBRegressor(base_score=.5, colsample_bylevel=1,
                    colsample_bytree=.7, gamma=0, learning_rate=.1,
                    max_delta_step=0, max_depth=7, min_child_weight=5,
                    missing=None, n_estimators=2500, silent=True,
                    subsample=.9)
        self.regLGB = LGBMRegressor(num_leaves=40,
                    boosting_type='gbdt', objective='regression',
                    learning_rate=.1, max_depth=-1,
                    n_estimators=2000, max_bin=250, silent=True,
                    reg_alpha=.001, reg_lambda=.01)

    def fit(self, X, y):
        self.regXGB.fit(X, y)
        self.regLGB.fit(X, y)

    def predict(self, X):
        XGB = self.regXGB.predict(X)
        LGB = self.regLGB.predict(X)
        return .7 * XGB + .3 * LGB
