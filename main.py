import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import data_preprocess,change_cat_type
from train import train_model
from test import test_model,test_loaded_model, loaded_model_pred
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
    print(f'Mean Absolute Error on Validation Set: {mae}')

    ########################################################Test on a single point##########################################################
    # #Write input like in the following order ['Car', 'Date Posted', 'Year', 'Mileage', 'Vehicle Type', 'Transmission','Wheel left/right', 'Color']
    # sample = ['Chevrolet Cruze', '13.02.2020', 2015, 30000.0, 'Sedan', 'Automatic','Left', 'Black']
    # X_test = pd.DataFrame([sample], columns = ['Car', 'Date Posted', 'Year', 'Mileage', 'Vehicle Type', 'Transmission','Wheel left/right', 'Color'])
    # X_test = change_cat_type(X_test)
    # X_train,X_test = data_preprocess(X_train, X_test)

    # model_path = 'models/your_model.json'
    # preds = loaded_model_pred(model_path, X_test)
    # print(f'Predition on Test Set: {preds}')

    ########################################################Test loaded model###############################################################
    # model_path = 'models/your_model.json'
    # mae = test_loaded_model(model_path, X_valid, y_valid)
    # print(f'Mean Absolute Error on Validation Set: {mae}')
