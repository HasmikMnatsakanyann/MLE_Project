import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from config import DataConfig

def preprocess_data(data):
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

    X = preprocessor.fit_transform(data.drop(DataConfig.TARGET_COL, axis=1))
    y = data[DataConfig.TARGET_COL].values

    return X, y
