import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime


# Load the trained model
model = joblib.load("best_model_rf.joblib")

# Judul aplikasi
st.title("🏘️ Prediksi Harga Rumah di Taipei")
st.image("data/taipei.jpg", use_container_width=True)
# Deskripsi aplikasi
st.write(
    "Aplikasi web ini dirancang untuk membantu pengguna, seperti calon pembeli, agen properti, dan pemilik real estate, dalam memprediksi harga rumah per unit area secara akurat di wilayah Taipei, Taiwan. Dengan memanfaatkan model Random Forest Regression yang telah dilatih menggunakan data historis properti, pengguna cukup memasukkan enam parameter penting yaitu tanggal transaksi, usia rumah, jarak ke MRT terdekat, jumlah minimarket terdekat, serta koordinat geografis (latitude dan longitude)."
)

# Form input dari pengguna
st.header("Masukkan Data Rumah")

col1, col2 = st.columns(2)

with col1:
    date_input = st.date_input("📅 Tanggal Transaksi")
    X2 = st.number_input(
        "🏠 Umur Rumah (tahun.bulan)",
        min_value=0.0,
        step=0.01,
        format="%.2f",
        help="Contoh: 10.50 = 10 tahun 6 bulan",
    )
    X3 = st.number_input("🚇 Jarak ke MRT Terdekat (meter)", min_value=0.0, step=1.0)

with col2:
    X4 = st.number_input(
        "🛒 Jumlah Minimarket Terdekat (walking distance)", min_value=0, step=1
    )
    X5 = st.number_input("📍 Latitude", format="%.6f")
    X6 = st.number_input("📍 Longitude", format="%.6f")


def date_to_decimal(date):
    year = date.year
    start_of_year = datetime.date(year, 1, 1)
    days_in_year = (datetime.date(year + 1, 1, 1) - start_of_year).days
    day_of_year = (date - start_of_year).days
    return year + day_of_year / days_in_year


def format_age(age):
    years = int(age)
    months = int(round((age - years) * 12))
    return f"{years} tahun {months} bulan"


def format_distance(dist):
    return f"{int(round(dist))} meter"


if st.button("🔍 Prediksi Harga"):
    formatted_date = date_input.strftime("%y/%m/%d")
    formatted_age = format_age(X2)
    formatted_distance = format_distance(X3)

    # Format jarak ke MRT jadi "xxx meter"
    formatted_distance = f"{int(X3)} meter"
    data_dict = {
        "Tanggal Transaksi": [formatted_date],
        "Umur Bangunan": [formatted_age],
        "Jarak ke MRT": [formatted_distance],
        "Jumlah Minimarket": [f"{X4} minimarket"],
        "Latitude": [f"{X5:.6f}°"],
        "Longitude": [f"{X6:.6f}°"],
    }
    st.subheader("Data yang Dimasukkan")
    st.table(pd.DataFrame(data_dict))

    input_data = np.array([[date_to_decimal(date_input), X2, X3, X4, X5, X6]])
    prediction = model.predict(input_data)[0]
    st.markdown(
        f"""
    ### 💰 Perkiraan Harga Rumah per Unit Area:
    <span style='color:green; font-size:24px; font-weight:bold;'>{prediction:.2f} New Taiwan Dollar/Ping</span>  
    (1 Ping = 3.3 meter persegi)
    """,
        unsafe_allow_html=True,
    )

    df_map = pd.DataFrame({"lat": [X5], "lon": [X6]})
    st.map(df_map)
