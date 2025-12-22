import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

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
# LOAD MODEL
# =================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "DELAY_MODEL_FINAL_CLOUD.pkl"

model = joblib.load(MODEL_PATH)

# =================================================
# HEADER
# =================================================
st.markdown("""
<div style="text-align:center;">
    <h1>📦 Amazon Delivery Delay Prediction</h1>
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
# MAKE DATAFRAME ACCORDING TO TRAINING MODEL
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

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(
            f"🚨 High Delay Risk — Probability: {prob:.2%}"
        )
    else:
        st.success(
            f"✅ Delivery On Time — Probability: {prob:.2%}"
        )

st.write("---")
st.markdown("<p style='text-align:center;color:#ccc;'>Machine Learning • Streamlit</p>", unsafe_allow_html=True)

