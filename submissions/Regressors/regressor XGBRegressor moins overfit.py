import xgboost as xgb
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = xgb.XGBRegressor(colsample_bylevel=0.8,
       colsample_bytree=0.55, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=7, min_child_weight=4, n_estimators=300,
       n_jobs=1, reg_lambda=0.001, subsample=0.95, objective='reg:squarederror')

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

import xgboost as xgb
from sklearn.base import BaseEstimator

#reg_alpha=0.01, reg_lambda=0,
class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = xgb.XGBRegressor(colsample_bylevel=0.8,
       colsample_bytree=0.55, gamma=0, learning_rate=0.3, max_delta_step=0,
       max_depth=7, min_child_weight=4, n_estimators=1000,
       n_jobs=1, reg_lambda=0.001, subsample=0.95, objective='reg:squarederror')

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)


        import xgboost as xgb
        from sklearn.base import BaseEstimator


        class Regressor(BaseEstimator):
            def __init__(self):
                self.reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=0.7, gamma=0, learning_rate=0.3, max_delta_step=0,
               max_depth=7, min_child_weight=5, missing=None, n_estimators=500,
               n_jobs=1, nthread=1, objective='reg:linear', random_state=0,
               reg_alpha=0.001, reg_lambda=0, seed=None,
               silent=True, subsample=0.8)

            def fit(self, X, y):
                self.reg.fit(X, y)

            def predict(self, X):
                return self.reg.predict(X)
