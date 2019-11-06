from sklearn.linear_model import ElasticNet

from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = ElasticNet(alpha=0.001, l1_ratio=0.7)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
