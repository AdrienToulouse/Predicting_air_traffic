import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor

from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.regXGB = XGBRegressor()
        self.regLGB = LGBMRegressor()
        self.regRdmF = RandomForestRegressor(n_estimators=50, max_depth=80, max_features=20)

        self.reg = LGBMRegressor()

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
        return self.reg.predict(prediction)
