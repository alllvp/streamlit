# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import joblib

# Чтение данных
housing = pd.read_csv('housing-deployment-reg.csv')

# Разделение на тренировочный и тестовый наборы
X = housing.drop(columns="SalePrice")
y = housing["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

# Создание пайплайна
pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        KNeighborsRegressor())

# Параметры для GridSearch
pipe_params = {
    'simpleimputer__strategy': ['median', 'mean'],
    'standardscaler__with_mean': [True, False],
    'kneighborsregressor__n_neighbors': list(range(1, 20)),
    'kneighborsregressor__weights': ['uniform', 'distance'],
    'kneighborsregressor__p': [1, 2],
    'kneighborsregressor__algorithm': ['ball_tree', 'kd_tree', 'brute']}

# GridSearchCV для поиска лучших параметров
trained_pipe = GridSearchCV(pipe, pipe_params, cv=5)
trained_pipe.fit(X_train, y_train)

# Тестирование модели на тестовом наборе
from sklearn.metrics import r2_score
y_pred = trained_pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Сохранение модели
joblib.dump(trained_pipe, 'trained_pipe_knn.sav')

