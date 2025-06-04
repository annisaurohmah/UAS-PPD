import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('rf_model.sav')

# Judul aplikasi
st.title("Prediksi Harga Rumah (Random Forest Regressor)")

# Form input dari pengguna
st.header("Masukkan Data Rumah")

X1 = st.number_input("X1 - Transaction Date (misal: 2013.25)", min_value=2000.0, max_value=2030.0, step=0.01)
X2 = st.number_input("X2 - House Age (tahun)", min_value=0.0, step=0.1)
X3 = st.number_input("X3 - Jarak ke MRT Terdekat (meter)", min_value=0.0, step=1.0)
X4 = st.number_input("X4 - Jumlah Convenience Store", min_value=0, step=1)
X5 = st.number_input("X5 - Latitude", format="%.6f")
X6 = st.number_input("X6 - Longitude", format="%.6f")

# Prediksi ketika tombol diklik
if st.button("Prediksi Harga"):
    input_data = np.array([[X1, X2, X3, X4, X5, X6]])
    prediction = model.predict(input_data)[0]
    st.success(f"Perkiraan harga rumah per unit area: {prediction:.2f}")
