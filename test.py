import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error

def test_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = model.predict(dtest)

    mae = mean_absolute_error(preds,y_test)
    return mae

def test_loaded_model(model_path, X_test, y_test):
    # Load the saved model
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_path)

    dtest = xgb.DMatrix(X_test, label=y_test)
    preds = loaded_model.predict(dtest)

    mae = mean_absolute_error(preds,y_test)
    return mae

def loaded_model_pred(model_path, X_test):
    # Load the saved model
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_path)

    dtest = xgb.DMatrix(X_test)
    preds = loaded_model.predict(dtest)

    return preds
