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
# CSS – Professional Theme
# =================================================
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #f59e0b;
}

/* Subtitle */
.sub-title {
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 25px;
}

/* Card style */
.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

/* Button */
.stButton > button {
    background-color: #f59e0b;
    color: #111827;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 700;
}
.stButton > button:hover {
    background-color: #fbbf24;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD PREPROCESSOR + MODEL
# =================================================
BASE_DIR = Path(__file__).resolve().parent.parent

preprocessor = joblib.load(BASE_DIR / "models" / "PREPROCESSOR.pkl")

xgb_model = xgb.Booster()
xgb_model.load_model(str(BASE_DIR / "models" / "XGBMODEL.json"))

# =================================================
# HEADER
# =================================================
st.markdown("""
<div style="text-align:center;">
    <div class="main-title">📦 Amazon Delivery Delay Predictor</div>
    <div class="sub-title">
        Predict whether an order will arrive <b>on time</b> or face a <b>delivery delay</b>
    </div>
</div>
""", unsafe_allow_html=True)

# =================================================
# SIDEBAR
# =================================================
st.sidebar.markdown("## 🧾 Order Details")

age = st.sidebar.slider("Agent Age", 18, 70, 30)
rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5, step=0.1)

weather = st.sidebar.selectbox("Weather Condition", ["Sunny", "Cloudy", "Stormy", "Sandstorms"])
traffic = st.sidebar.selectbox("Traffic Level", ["Low", "Medium", "High", "Jam"])
vehicle = st.sidebar.selectbox("Delivery Vehicle", ["motorcycle", "scooter"])
area = st.sidebar.selectbox("Delivery Area", ["Urban", "Metropolitian", "Rural"])
category = st.sidebar.selectbox("Product Category", ["Clothing", "Electronics", "Sports", "Cosmetics", "Toys"])

# =================================================
# INPUT DATAFRAME
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
# MAIN CARD
# =================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("### 📊 Prediction Result")

if st.button("🚀 Predict Delivery Status", use_container_width=True):

    X = preprocessor.transform(input_df)
    dtest = xgb.DMatrix(X)

    proba = float(xgb_model.predict(dtest)[0])
    confidence = round(proba * 100, 2)

    if proba >= 0.5:
        st.error(f"🚨 **High Risk of Delay**")
        st.write(f"Prediction Confidence: **{confidence}%**")
        st.progress(confidence / 100)
    else:
        st.success(f"✅ **On-Time Delivery Expected**")
        st.write(f"Prediction Confidence: **{confidence}%**")
        st.progress(confidence / 100)

st.markdown("</div>", unsafe_allow_html=True)

# =================================================
# FOOTER
# =================================================
st.markdown("""
<div class="footer">
Built with ❤️ using <b>Python, XGBoost & Streamlit</b><br>
© 2025 – Amazon Delivery Delay Prediction Project
</div>
""", unsafe_allow_html=True)

