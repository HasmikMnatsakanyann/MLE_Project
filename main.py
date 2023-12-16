import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data
from train import train_model
from test import test_model
from config import DataConfig

if __name__ == "__main__":
    # Load your data
    data_path = "ARM_Cars.csv"
    data = pd.read_csv(data_path)

    # Data preprocessing
    X, y = preprocess_data(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=DataConfig.RANDOM_SEED)

    # Train the model
    model = train_model(X_train, y_train)

    # Test the model
    mae = test_model(model, X_test, y_test)
    print(f'Mean Absolute Error on Test Set: {mae}')
