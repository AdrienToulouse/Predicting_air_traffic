import xgboost as xgb
from sklearn.base import BaseEstimator

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
