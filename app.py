import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('best_model_rf.joblib')

# Judul aplikasi
st.title("Prediksi Harga Rumah (Random Forest Regressor)")
st.image("None", use_column_width=True)
st.write("Aplikasi ini memprediksi harga rumah berdasarkan beberapa fitur seperti tanggal transaksi, usia rumah, jarak ke MRT, jumlah convenience store, dan koordinat geografis (latitude dan longitude).")

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
    data_dict = {
    "X1 - Transaction Date": [X1],
    "X2 - House Age": [X2],
    "X3 - Distance to MRT": [X3],
    "X4 - Convenience Stores": [X4],
    "X5 - Latitude": [X5],
    "X6 - Longitude": [X6]
}
    st.subheader("Data yang Dimasukkan")
    st.table(pd.DataFrame(data_dict))

    input_data = np.array([[X1, X2, X3, X4, X5, X6]])
    prediction = model.predict(input_data)[0]
    st.success(f"Perkiraan harga rumah per unit area: {prediction:.2f}")

    df_map = pd.DataFrame({'lat': [X5], 'lon': [X6]})
    st.map(df_map)
