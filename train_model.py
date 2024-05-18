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
import joblib

# Loading data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

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

# Filling in the blanks
train_df = fill_missing_data(train_df)
test_df = fill_missing_data(test_df)

# Preparing training data
X_train_raw = train_df.drop(columns='SalePrice')
y_train = train_df['SalePrice']

# Transforming training data
X_train, preprocessor = preprocess_data(X_train_raw, fit=True)

# Logarithm the target variable
y_train_log = np.log(y_train)

# Defining a pipeline with SelectKBest and XGBoost model
pipeline = Pipeline([
    ('select', SelectKBest(score_func=mutual_info_regression)),
    ('model', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Options for RandomizedSearchCV
param_grid = {
    'select__k': range(10, X_train.shape[1] + 1),
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': range(3, 10),
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0]
}

# Definition of the evaluation function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Grid search with cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = RandomizedSearchCV(pipeline, param_grid, cv=kf, scoring=rmse_scorer, n_iter=100, verbose=1, random_state=42)
grid_search.fit(X_train, y_train_log)

# Best model parameters
best_params = grid_search.best_params_
print(f"Лучшие параметры модели: {best_params}")

# Using the best parameters for the model
best_model = grid_search.best_estimator_

# Saving the model and preprocessor
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
