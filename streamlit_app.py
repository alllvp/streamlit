# streamlit_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Определение класса ManhattanDistance, если он не может быть импортирован
class ManhattanDistance:
    def __call__(self, x, y):
        return np.sum(np.abs(x - y), axis=1)

# Загрузка модели
model = joblib.load('trained_pipe_knn.sav')

# Заголовок для приложения
st.title('Housing Price Prediction')

# Форма для ввода данных
LotArea = st.number_input("Lot Area", min_value=0)
TotalBsmtSF = st.number_input("Basement Square Feet", min_value=0)
BedroomAbvGr = st.number_input("Number of Bedrooms", min_value=0)
GarageCars = st.number_input("Car spaces in Garage", min_value=0)

# Создание DataFrame с введенными пользователем данными
new_house = pd.DataFrame({
    'LotArea': [LotArea],
    'TotalBsmtSF': [TotalBsmtSF],
    'BedroomAbvGr': [BedroomAbvGr],
    'GarageCars': [GarageCars]
})

# Предсказание
if st.button('Predict'):
    prediction = model.predict(new_house)
    st.write("The price of the house is:", prediction[0])
