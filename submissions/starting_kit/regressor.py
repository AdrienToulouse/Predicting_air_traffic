import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.regXGB = XGBRegressor(base_score=0.5, colsample_bylevel=1,
               colsample_bytree=0.7, gamma=0, learning_rate=0.16, max_delta_step=0,
               max_depth=7, min_child_weight=5, missing=None, n_estimators=2450,
               silent=True, subsample=.9, n_jobs=4)
        self.regLGB = LGBMRegressor(num_leaves=40,
                    boosting_type='gbdt',
                    objective='regression',
                    learning_rate=0.1,
                    max_depth=-1,
                    n_estimators=2000,
                    max_bin=200, silent=True,
                    reg_alpha=.001,
                    reg_lambda=.01,
                    n_jobs=4)
        self.reg = KernelRidge(alpha=.01, kernel='polynomial', degree=3, gamma=1)

    def fit(self, X, y):
        self.regXGB.fit(X, y)
        self.regLGB.fit(X, y)

        XGB = self.regXGB.predict(X)
        LGB = self.regLGB.predict(X)

        prediction = pd.DataFrame({'XGB': XGB, 'LGB': LGB})
        self.reg.fit(prediction, y)


    def predict(self, X):
        XGB = self.regXGB.predict(X)
        LGB = self.regLGB.predict(X)

        prediction = pd.DataFrame({'XGB': XGB, 'LGB': LGB})
        stack_pred = self.reg.predict(prediction)
        return stack_pred
