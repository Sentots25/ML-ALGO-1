import streamlit as st
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title aplikasi
st.title("Prediksi Keesetaraan Gender menggunakan Machine Learning")

# Input fitur
sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

# Tombol prediksi
if st.button("Prediksi"):
    # Masukkan data input ke model
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    target_names = ["Setosa", "Versicolor", "Virginica"]
    st.write(f"Hasil prediksi: {target_names[prediction]}")
