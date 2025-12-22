import streamlit as st
import pandas as pd
from pathlib import Path
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
# ULTRA PREMIUM CSS
# =================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    color: white;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #232526, #414345);
    padding: 20px;
}
[data-testid="stSidebar"] label {
    color: #F9FAFB !important;
}
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF9900, #FFD194);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    color: #E5E7EB;
    font-size: 1.1rem;
}
.card {
    background: linear-gradient(145deg, #ffffff, #f1f5f9);
    color: #111827;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.35);
    margin-bottom: 30px;
}
.metric-title {
    color: #6B7280;
    font-size: 1rem;
}
.metric-value {
    font-size: 2.6rem;
    font-weight: 700;
}
.stButton > button {
    width: 100%;
    height: 65px;
    font-size: 1.3rem;
    font-weight: 700;
    border-radius: 14px;
    background: linear-gradient(90deg, #FF8008, #FFC837);
    color: black;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #FFC837, #FF8008);
}
.success {
    background: linear-gradient(135deg, #11998E, #38EF7D);
    padding: 30px;
    border-radius: 18px;
    font-size: 1.2rem;
}
.danger {
    background: linear-gradient(135deg, #CB356B, #BD3F32);
    padding: 30px;
    border-radius: 18px;
    font-size: 1.2rem;
}
.footer {
    text-align: center;
    margin-top: 60px;
    color: #CBD5E1;
}

</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD BOOSTER MODEL
# =================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "DELAY_MODEL_FINAL_BOOSTER.json"

model = xgb.Booster()
model.load_model(MODEL_PATH)

# =================================================
# HEADER
# =================================================
st.markdown("""
<div style="text-align:center;">
    <div class="main-title">📦 Amazon Delivery Delay Prediction</div>
    <p class="subtitle">
        AI-powered system to predict delivery delays before they happen
    </p>
</div>
""", unsafe_allow_html=True)

# =================================================
# SIDEBAR INPUTS
# =================================================
st.sidebar.markdown("## 📝 Order Details")

age = st.sidebar.slider("Agent Age", 18, 70, 30)
rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5, step=0.1)
weather = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Stormy", "Sandstorms"])
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
vehicle = st.sidebar.selectbox("Vehicle", ["motorcycle", "scooter"])
area = st.sidebar.selectbox("Area", ["Urban", "Metropolitian", "Rural"])
category = st.sidebar.selectbox("Category", ["Clothing", "Electronics", "Sports", "Cosmetics", "Toys"])
duration = st.sidebar.number_input("Estimated Doorstep Delivery Time (minutes)", 10, 300, 120)

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
    "Category": category,
    "Duration": duration
}])

# =================================================
# MAIN METRICS DISPLAY
# =================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Agent Rating</div>
            <div class="metric-value">{rating}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">Estimated Duration</div>
            <div class="metric-value">{duration} mins</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =================================================
# PREDICTION
# =================================================
if st.button("🚀 Predict Delivery Status"):

    # Convert to DMatrix
    dtest = xgb.DMatrix(input_df)

    # Booster prediction returns probability
    proba = float(model.predict(dtest)[0])

    pred = 1 if proba >= 0.5 else 0

    if pred == 1:
        st.markdown(
            f"""
            <div class="danger">
                🚨 <b>High Risk of Delay</b><br><br>
                Probability of delay: <b>{proba:.2%}</b><br>
                • Monitor traffic closely<br>
                • Assign senior agent<br>
                • Proactive customer alert
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="success">
                ✅ <b>Delivery Likely On Time</b><br><br>
                Probability of delay: <b>{proba:.2%}</b><br>
                • Smooth delivery expected
            </div>
            """,
            unsafe_allow_html=True
        )

# =================================================
# FOOTER
# =================================================
st.markdown(
    """
    <div class="footer">
    Built with Python • XGBoost • Streamlit • Real-world Logistics Data
    </div>
    """,
    unsafe_allow_html=True
)

