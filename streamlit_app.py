import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузка модели и предобработчика
best_model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Категориальные признаки для Ordinal Encoding
ordinal_features = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
    'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
]

# Определение порядка категорий для OrdinalEncoder
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

# Категориальные признаки для One-Hot Encoding
onehot_features = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
    'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 
    'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'Functional', 
    'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 
    'SaleType', 'SaleCondition'
]

# Числовые признаки
numeric_features = [
    'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 
    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]

# Предобработка данных для предсказания
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    df = fill_missing_data(df)
    X_transformed = preprocess_data(df, preprocessor=preprocessor, fit=False)
    return X_transformed

def main():
    st.title("House Price Prediction")

    input_data = {}
    for feature in numeric_features:
        input_data[feature] = st.number_input(feature, value=0.0)
    
    for feature in ordinal_features:
        input_data[feature] = st.selectbox(feature, ordinal_categories[ordinal_features.index(feature)])
    
    for feature in onehot_features:
        input_data[feature] = st.selectbox(feature, ['missing', *df[feature].unique()])

    if st.button("Predict"):
        X_input = preprocess_input(input_data)
        prediction_log = best_model.predict(X_input)
        prediction = np.exp(prediction_log)
        st.write("The predicted house price is:", prediction[0])

if __name__ == '__main__':
    main()
