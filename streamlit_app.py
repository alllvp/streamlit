import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Loading the model and preprocessor
best_model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Categorical features for Ordinal Encoding
ordinal_features = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
    'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
]

# Determining the order of categories for OrdinalEncoder
ordinal_categories = [
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # ExterQual
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # ExterCond
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # BsmtQual
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # BsmtCond
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # HeatingQC
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # KitchenQual
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # FireplaceQu
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # GarageQual
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # GarageCond
    ['Fa', 'TA', 'Gd', 'Ex']  # PoolQC
]

# Determining the order of categories for One-Hot Encoding
onehot_features = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
    'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 
    'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 
    'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 
    'SaleType', 'SaleCondition'
]

# Numerical characteristics
numeric_features = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]

# Data preprocessing for prediction
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    df = fill_missing_data(df)
    X_transformed = preprocess_data(df, preprocessor=preprocessor, fit=False)
    return X_transformed

# Function to fill in the blanks
def fill_missing_data(df):
    all_features = numeric_features + ordinal_features + onehot_features
    missing_columns = [col for col in all_features if col not in df.columns]
    if missing_columns:
        print(f"Warning: Columns not found in data: {missing_columns}")

    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in ordinal_features + onehot_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

# Data preprocessing
def preprocess_data(X, preprocessor=None, fit=False):
    imputer_cat = SimpleImputer(strategy='constant', fill_value='missing')
    imputer_num = SimpleImputer(strategy='constant', fill_value=0)

    numeric_transformer = Pipeline(steps=[
        ('imputer', imputer_num),
        ('scaler', StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', imputer_cat),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    onehot_transformer = Pipeline(steps=[
        ('imputer', imputer_cat),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('ord', ordinal_transformer, ordinal_features),
                ('onehot', onehot_transformer, onehot_features)
            ]
        )

    if fit:
        X_transformed = preprocessor.fit_transform(X)
        return X_transformed, preprocessor
    else:
        X_transformed = preprocessor.transform(X)
        return X_transformed

def main():
    st.title("House Price Prediction")

    input_data = {}
    for feature in numeric_features:
        input_data[feature] = st.number_input(feature, value=0.0)
    
    for feature, categories in zip(ordinal_features, ordinal_categories):
        input_data[feature] = st.selectbox(feature, categories)
    
    for feature in onehot_features:
        # Используем фиксированные значения для выбора
        input_data[feature] = st.selectbox(feature, ['missing', 'Option1', 'Option2'])

    if st.button("Predict"):
        X_input = preprocess_input(input_data)
        prediction_log = best_model.predict(X_input)
        prediction = np.exp(prediction_log)
        st.write("The predicted house price is:", prediction[0])

if __name__ == '__main__':
    main()
