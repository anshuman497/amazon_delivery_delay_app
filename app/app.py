import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import xgboost as xgb
import numpy as np

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Amazon Delivery Delay Predictor",
    page_icon="📦",
    layout="wide"
)

# =================================================
# CSS
# =================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD PREPROCESSOR + MODEL
# =================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Load preprocessing pipeline
preprocessor = joblib.load(BASE_DIR / "models" / "PREPROCESSOR.pkl")

# Load trained XGBoost booster model
xgb_model = xgb.Booster()
xgb_model.load_model(str(BASE_DIR / "models" / "XGBMODEL.json"))

# =================================================
# HEADER
# =================================================
st.markdown("""
<div style="text-align:center;">
    <h1>📦 Amazon Delivery Delay Prediction</h1>
    <p>Machine learning model that predicts whether an order will arrive late or on time.</p>
</div>
""", unsafe_allow_html=True)

# =================================================
# SIDEBAR INPUTS
# =================================================
st.sidebar.header("📝 Enter Order Details")

age = st.sidebar.slider("Agent Age", 18, 70, 30)
rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5, step=0.1)
weather = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Stormy", "Sandstorms"])
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
vehicle = st.sidebar.selectbox("Vehicle", ["motorcycle", "scooter"])
area = st.sidebar.selectbox("Area", ["Urban", "Metropolitian", "Rural"])
category = st.sidebar.selectbox("Category", ["Clothing", "Electronics", "Sports", "Cosmetics", "Toys"])

# =================================================
# CREATE DATAFRAME
# =================================================
input_df = pd.DataFrame([{
    "Agent_Age": age,
    "Agent_Rating": rating,
    "Weather": weather,
    "Traffic": traffic,
    "Vehicle": vehicle,
    "Area": area,
    "Category": category
}])

# =================================================
# PREDICT BUTTON
# =================================================
if st.button("🚀 Predict Delivery Status"):

    # 1️⃣ preprocess input data
    X = preprocessor.transform(input_df)

    # 2️⃣ convert to XGBoost format
    dtest = xgb.DMatrix(X)

    # 3️⃣ get probability score
    proba = float(xgb_model.predict(dtest)[0])

    pred = 1 if proba >= 0.5 else 0

    if pred == 1:
        st.error(f"🚨 High Delay Risk — Probability: {proba:.2%}")
    else:
        st.success(f"✅ On Time Delivery — Probability: {proba:.2%}")

# =================================================
# FOOTER
# =================================================
st.write("---")
st.markdown(
    "<p style='text-align:center;color:#ccc;'>Built with XGBoost + Streamlit</p>",
    unsafe_allow_html=True
)

