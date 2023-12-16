import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from config import DataConfig


def data_preprocess(train_data,valid_data):
    train_data['Date Posted'] = pd.to_datetime(train_data['Date Posted'], format='%d.%m.%Y')
    train_data['Date Posted'] = train_data['Date Posted'].dt.year
    train_data['Age'] = train_data['Date Posted'] - train_data['Year']
    train_data.drop(['Year', 'Date Posted'], axis=1, inplace=True)

    valid_data['Date Posted'] = pd.to_datetime(valid_data['Date Posted'], format='%d.%m.%Y')
    valid_data['Date Posted'] = valid_data['Date Posted'].dt.year
    valid_data['Age'] = valid_data['Date Posted'] - valid_data['Year']
    valid_data.drop(['Year', 'Date Posted'], axis=1, inplace=True)

    # Selecting columns for one-hot encoding and standard scaling
    categorical_cols = ['Car', 'Vehicle Type', 'Wheel left/right', 'Color']
    numerical_cols = ['Mileage', 'Age']
    categorical_col = ['Transmission']

    # Creating transformers for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat_cols', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('cat_col', OrdinalEncoder(categories=[['Semi-Automatic', 'Variator','Manual', 'Automatic']]), categorical_col)
        ]
    )

    X_train = preprocessor.fit_transform(train_data)
    X_train = np.array(X_train.todense())

    X_valid = preprocessor.transform(valid_data)
    X_valid = np.array(X_valid.todense())

    return X_train, X_valid

def train_data_preprocess(data):
    data['Date Posted'] = pd.to_datetime(data['Date Posted'], format='%d.%m.%Y')
    data['Date Posted'] = data['Date Posted'].dt.year
    data['Age'] = data['Date Posted'] - data['Year']
    data.drop(['Year', 'Date Posted'], axis=1, inplace=True)

    # Selecting columns for one-hot encoding and standard scaling
    categorical_cols = ['Car', 'Vehicle Type', 'Wheel left/right', 'Color']
    numerical_cols = ['Mileage', 'Age']
    categorical_col = ['Transmission']

    # Creating transformers for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat_cols', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('cat_col', OrdinalEncoder(categories=[['Semi-Automatic', 'Variator','Manual', 'Automatic']]), categorical_col)
        ]
    )

    X = preprocessor.fit_transform(data)
    X = np.array(X.todense())

    return X, preprocessor

def valid_data_preprocess(data, train_data):
    data['Date Posted'] = pd.to_datetime(data['Date Posted'], format='%d.%m.%Y')
    data['Date Posted'] = data['Date Posted'].dt.year
    data['Age'] = data['Date Posted'] - data['Year']
    data.drop(['Year', 'Date Posted'], axis=1, inplace=True)

    # Selecting columns for one-hot encoding and standard scaling
    categorical_cols = ['Car', 'Vehicle Type', 'Wheel left/right', 'Color']
    numerical_cols = ['Mileage', 'Age']
    categorical_col = ['Transmission']

    # Creating transformers for preprocessing
    _, preprocessor = train_data_preprocess(train_data)

    X = preprocessor.transform(data)
    X = np.array(X.todense()) 

    return X
