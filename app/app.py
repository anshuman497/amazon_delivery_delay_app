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

</style>
""", unsafe_allow_html=True)

# =================================================
# LOAD PIPELINE MODEL (.pkl)
# =================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "DELAY_MODEL_FINAL.pkl"

model = joblib.load(MODEL_PATH)

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
st.sidebar.header("📝 Enter Order Details")

age = st.sidebar.slider("Agent Age", 18, 70, 30)
rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5, step=0.1)
weather = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Stormy", "Sandstorms"])
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
vehicle = st.sidebar.selectbox("Vehicle", ["motorcycle", "scooter"])
area = st.sidebar.selectbox("Area", ["Urban", "Metropolitian", "Rural"])
category = st.sidebar.selectbox("Category", ["Clothing", "Electronics", "Sports", "Cosmetics", "Toys"])
duration = st.sidebar.number_input("Estimated Delivery Time (minutes)", 10, 300, 120)

# =================================================
# FORM DATAFRAME
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
# SHOW CARDS
# =================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Agent Rating")
    st.write(f"**{rating} / 5.0**")

with col2:
    st.subheader("⏱ Estimated Time")
    st.write(f"**{duration} minutes**")

# =================================================
# PREDICT BUTTON
# =================================================
if st.button("🚀 Predict Delivery Status"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(
            f"""
            🚨 **High Risk of Delay**  
            Probability of delay: **{prob:.2%}**
            """
        )
    else:
        st.success(
            f"""
            ✅ **Delivery Expected On Time**  
            Probability of delay: **{prob:.2%}**
            """
        )

# =================================================
# FOOTER
# =================================================
st.write("---")
st.markdown(
    "<p style='text-align:center;color:#ccc;'>Built with Python • Machine Learning • Streamlit</p>",
    unsafe_allow_html=True
)

