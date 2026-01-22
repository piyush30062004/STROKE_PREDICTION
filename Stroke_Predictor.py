import streamlit as st
import pandas as pd
import joblib

# ========== CUSTOM CSS FOR BEAUTIFUL UI ==========
st.markdown(
    """
    <style>
    .main {
        background-color: #eef2f3;
        padding: 20px;
    }

    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        padding: 10px;
    }

    .subtitle {
        text-align: center;
        color: #34495e;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .result-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        margin-top: 25px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ========== TITLE UI ==========
st.markdown('<p class="title">🧠 Stroke Prediction Web App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient details below to predict stroke risk.</p>', unsafe_allow_html=True)

# ========== LOAD MODEL ==========
model = joblib.load("model.pkl")

# ========== USER INPUTS ==========
st.subheader("Patient Information")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["children", "Govt_job", "Never_worked", "Private", "Self-employed"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=90.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

# ========== PREPARE INPUT ==========
input_data = pd.DataFrame({
    'gender': [gender],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [Residence_type],
    'smoking_status': [smoking_status],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi]
})

# ========== PREDICT BUTTON ==========
if st.button("Predict Stroke Risk"):

    # Get probability
    proba = model.predict_proba(input_data)[0][1] * 100

    # Define risk category
    if proba < 20:
        status = "🟢 Very Unlikely"
    elif 20 <= proba < 40:
        status = "🟡 Unlikely"
    elif 40 <= proba < 60:
        status = "🟠 Moderate Risk"
    elif 60 <= proba < 80:
        status = "🟤 Likely"
    else:
        status = "🔴 Highly Likely"

    # Result Box UI
    st.markdown(
        f"""
        <div class="result-box">
            <h2>🔍 Stroke Risk Prediction</h2>
            <h3>Probability of Stroke: <b>{proba:.2f}%</b></h3>
            <h3>Risk Category: <b>{status}</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )
