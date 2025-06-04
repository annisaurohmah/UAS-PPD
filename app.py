import streamlit as st
import joblib
import numpy as np
import pandas as pd



# Load the trained model
model = joblib.load('rf_model.sav')

# Judul aplikasi
st.title("Prediksi Harga Rumah di Taipei")
st.image("data/taipei.jpg", use_container_width=True)
# Deskripsi aplikasi
st.write("Aplikasi ini memprediksi harga rumah berdasarkan beberapa fitur seperti tanggal transaksi, usia rumah, jarak ke MRT, jumlah convenience store, dan koordinat geografis (latitude dan longitude).")

# Form input dari pengguna
st.header("Masukkan Data Rumah")

import datetime

# 1. Ambil input tanggal dari user
date_input = st.date_input("Tanggal Transaksi")

# 2. Ubah ke format desimal tahun (contoh: 2013.25)
def date_to_decimal(date):
    year = date.year
    start_of_year = datetime.date(year, 1, 1)
    days_in_year = (datetime.date(year + 1, 1, 1) - start_of_year).days
    day_of_year = (date - start_of_year).days
    return year + day_of_year / days_in_year

X1 = date_to_decimal(date_input)
X2 = st.number_input("Umur Rumah (tahun)", min_value=0.0, step=0.1)
X3 = st.number_input("Jarak ke MRT Terdekat (meter)", min_value=0.0, step=1.0)
X4 = st.number_input("Jumlah Convenience Store", min_value=0, step=1)
X5 = st.number_input("Latitude", format="%.6f")
X6 = st.number_input("Longitude", format="%.6f")

# Prediksi ketika tombol diklik


if st.button("Prediksi Harga"):
    data_dict = {
    "Tanggal Transaksi": [X1],
    "Umur Bangunan": [X2],
    "Jarak ke MRT": [X3],
    "Jumlah Convenience Stores": [X4],
    "Latitude": [X5],
    "Longitude": [X6]
}
    st.subheader("Data yang Dimasukkan")
    st.table(pd.DataFrame(data_dict))

    input_data = np.array([[X1, X2, X3, X4, X5, X6]])
    prediction = model.predict(input_data)[0]
    st.success(f"Perkiraan harga rumah per unit area: {prediction:.2f}")

    df_map = pd.DataFrame({'lat': [X5], 'lon': [X6]})
    st.map(df_map)
