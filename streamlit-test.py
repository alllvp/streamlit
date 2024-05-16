import streamlit as st
import pickle
import pandas as pd

# Загрузка модели
model = pickle.load(open('trained_pipe_knn.sav', 'rb'))

st.title("Housing Prices Prediction")

st.write("""
### Project description
We have trained several models to predict the price of a house based on features such as the area of the house and the condition and quality of their different rooms.
""")

# Ввод данных пользователем
LotArea = st.number_input("Lot Area", min_value=0)
TotalBsmtSF = st.number_input("Basement Square Feet", min_value=0)
BedroomAbvGr = st.number_input("Number of Bedrooms", min_value=0)
GarageCars = st.number_input("Car spaces in Garage", min_value=0)

# Формирование данных для предсказания
new_house = pd.DataFrame({
    'LotArea': [LotArea],
    'TotalBsmtSF': [TotalBsmtSF],
    'BedroomAbvGr': [BedroomAbvGr],
    'GarageCars': [GarageCars]
})

# Предсказание
if st.button("Predict"):
    prediction = model.predict(new_house)
    st.write("The predicted price of the house is:", prediction[0])
