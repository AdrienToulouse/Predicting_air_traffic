from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=100, min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', n_jobs=4)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
