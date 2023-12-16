import xgboost as xgb
from config import ModelConfig

def train_model(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(ModelConfig.XGB_PARAMS, dtrain, ModelConfig.num_rounds)

    return bst
