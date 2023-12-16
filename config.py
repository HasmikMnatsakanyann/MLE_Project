class ModelConfig:
    XGB_PARAMS = {
        'max_depth': 8,
        'eta': 0.09,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'lambda': 2,
        'alpha': 2,
        'objective': 'reg:squarederror'
    }

class DataConfig:
    FEATURE_COLS = ['Mileage', 'Age', 'Car', 'Vehicle Type', 'Wheel left/right', 'Color', 'Transmission']
    TARGET_COL = 'Price ($)'
    RANDOM_SEED = 8
