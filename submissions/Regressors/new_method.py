from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.regXGB = XGBRegressor(base_score=0.5, colsample_bylevel=1,
               colsample_bytree=0.7, gamma=0, learning_rate=0.16, max_delta_step=0,
               max_depth=7, min_child_weight=5, missing=None, n_estimators=2450,
               silent=True, subsample=.9)
        self.regLGB = LGBMRegressor(num_leaves=20,
                    boosting_type='gbdt',
                    objective='regression',
                    learning_rate=0.1,
                    max_depth=-1,
                    n_estimators=1800,
                    bagging_fraction=0.52,
                    feature_fraction=0.63,
                    max_bin=255, silent=True,silent=True,
                    reg_alpha=.001,
                    reg_lambda=.01)
        #self.regRdmF = RandomForestRegressor(n_estimators=50, max_depth=80, max_features=20)

    def fit(self, X, y):
        self.regXGB.fit(X, y)
        self.regLGB.fit(X, y)
        #self.regRdmF.fit(X, y)

    def predict(self, X):
        XGB = self.regXGB.predict(X)
        LGB = self.regLGB.predict(X)
        #RdmF = self.regRdmF.predict(X)

        predict = XGB * 0.6 + LGB * 0.4

        return predict


X_values = X.values
    lalaM = LassoLars(alpha=0.000037)
    base_models = []
    base_models.append(regXGB)
    base_models.append(regLGB)
    base_models.append(regRdmF)
    meta_model = lalaM
    kf_predictions = np.zeros((X.shape[0], len(base_models)))
    for i, model in enumerate(base_models):
        for train_index ,test_index in stack_kfold.split(X_values):
            model_pred = model.predict(X_values[test_index])
            kf_predictions[test_index, i] = model_pred

    #teach the meta model
    meta_model.fit(kf_predictions, Y_values)

    preds = []

    for model in base_models:
        pred = model.predict(X)
        preds.append(pred)
    base_predictions = np.column_stack(preds)

    #get stacked prediction
    stacked_predict = meta_model.predict(base_predictions)

    def getPred(model, X, Y, test):
        model.fit(X, Y)
        pred = model.predict(test)
        return pred

    def getCSV(file_name, id_col, pred):
        sub = pd.DataFrame()
        sub['Id'] = id_col
        sub['SalePrice'] = np.expm1(pred)
        sub.to_csv(file_name,index=False)

    #get other tuned models prediction
    xgb_pred =  getPred(xgbM, X_learning, Y_learning, X_test)
    eln_pred =  getPred(elnM, X_learning, Y_learning, X_test)
    rid_pred =  getPred(ridM, X_learning, Y_learning, X_test)

    #percentage is based on each model's CV scores
