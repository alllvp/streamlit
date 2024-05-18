import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.impute import SimpleImputer

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

# Функция для заполнения пропусков
def fill_missing_data(df):
    all_features = numeric_features + ordinal_features + onehot_features
    missing_columns = [col for col in all_features if col not in df.columns]
    if missing_columns:
        st.warning(f"Columns not found in data: {missing_columns}")

    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in ordinal_features + onehot_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

# Предобработка данных
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

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data uploaded successfully")

        # Заполнение пропусков
        df = fill_missing_data(df)

        # Ввод значений для предсказания
        input_data = {}
        for feature in numeric_features:
            if feature in df.columns:
                input_data[feature] = st.number_input(feature, value=float(df[feature].median()))

        for feature in ordinal_features:
            if feature in df.columns:
                input_data[feature] = st.selectbox(feature, ordinal_categories[ordinal_features.index(feature)])

        for feature in onehot_features:
            if feature in df.columns:
                input_data[feature] = st.selectbox(feature, df[feature].unique())

        input_df = pd.DataFrame([input_data])
        X_input, _ = preprocess_data(input_df, preprocessor=preprocessor, fit=False)

        if st.button("Predict"):
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model.load_model('trained_pipe_knn.json')  # Загрузка сохраненной модели
            prediction_log = model.predict(X_input)
            prediction = np.exp(prediction_log)
            st.write("The predicted house price is:", prediction[0])

if __name__ == '__main__':
    main()
