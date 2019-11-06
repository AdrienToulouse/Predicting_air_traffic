from sklearn.linear_model import Lasso


from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = Lasso(alpha=1e-4, normalize=True, max_iter=1e5)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)
