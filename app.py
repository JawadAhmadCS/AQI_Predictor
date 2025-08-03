import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px

# Title
st.set_page_config(page_title="AQI Predictor", layout="centered")
st.title("🌫️ AQI Prediction Dashboard - Islamabad")

# --- Load latest weather data ---
weather_file = "data/weather_init.csv"
if os.path.exists(weather_file):
    df_weather = pd.read_csv(weather_file)
    latest = df_weather.iloc[-1]
    
    st.subheader("Current Weather")
    st.markdown(f"**Timestamp:** {latest['timestamp']}")
    st.markdown(f"**AQI (US):** `{latest['aqius']}`")
    st.markdown(f"**Temperature:** {latest['temperature']}°C")
    st.markdown(f"**Humidity:** {latest['humidity']}%")
    st.markdown(f"**Wind Speed:** {latest['wind_speed']} m/s`")
else:
    st.error("No weather data found.")

# --- Load AQI Trend ---
feature_file = "data/features.csv"
if os.path.exists(feature_file):
    df_feat = pd.read_csv(feature_file)
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
    st.subheader("AQI Trend (Past Hours)")
    fig = px.line(df_feat, x="timestamp", y="aqius", title="AQI Over Time")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Feature data not available yet.")

# --- Predict next 3 hours ---
model_path = "models/aqi_model.pkl"
if os.path.exists(model_path) and os.path.exists(feature_file):
    model = joblib.load(model_path)
    last_row = df_feat.dropna().iloc[-1:]

    X = last_row[["hour", "day", "month", "weekday", "aqius", "aqius_change"]]

    predictions = []
    for i in range(3):
        pred = model.predict(X)[0]
        predictions.append(round(pred, 2))
        X["aqius_change"] = pred - X["aqius"].values[0]
        X["aqius"] = pred
        X["hour"] = (X["hour"] + 1) % 24

    st.subheader("AQI Prediction (Next 3 Hours)")
    for i, p in enumerate(predictions, 1):
        st.markdown(f"**Hour +{i}:** `{p}` AQI")
else:
    st.warning("Model not trained yet or feature data missing.")
