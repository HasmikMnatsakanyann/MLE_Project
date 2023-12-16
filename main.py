import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import data_preprocess
from train import train_model
from test import test_model
from config import DataConfig

if __name__ == "__main__":
    # Load your data
    data_path = "ARM_Cars.csv"
    data = pd.read_csv(data_path)

    X = data.drop(DataConfig.TARGET_COL,axis=1)
    y = data[DataConfig.TARGET_COL]

    # Split data into train and test sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=DataConfig.RANDOM_SEED)
    
    X_train = pd.DataFrame(X_train, columns = data.drop('Price ($)',axis=1).columns)
    X_valid = pd.DataFrame(X_valid, columns = data.drop('Price ($)',axis=1).columns)

    # Data preprocessing
    X_train, X_valid = data_preprocess(X_train,X_valid)

    # Train the model
    model = train_model(X_train, y_train)

    # Test the model
    mae = test_model(model, X_valid, y_valid)
    print(f'Mean Absolute Error on Test Set: {mae}')
