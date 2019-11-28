import xgboost as xgb
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                        colsample_bylevel=1,
                        learning_rate=0.16, min_split_loss=0,
                        max_depth=4, n_estimators=1000, min_child_weight=1,
                        n_jobs=1, nthread=1, objective='reg:squarederror')


    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
