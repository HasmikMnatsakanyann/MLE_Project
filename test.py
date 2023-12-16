import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error

def test_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = model.predict(dtest)

    mae = mean_absolute_error(preds,y_test)
    return mae
