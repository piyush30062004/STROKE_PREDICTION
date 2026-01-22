import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="NeuroGuard Elite", layout="wide", initial_sidebar_state="collapsed")

# ================= DARK MODE VISIBILITY CSS =================
st.markdown("""
<style>
    /* 1. RESTORE DARK GRADIENT BACKGROUND */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        background-attachment: fixed;
    }

    /* 2. MAKE ALL FORM LABELS PURE WHITE & BOLD */
    label, .stMarkdown p, .stSelectbox label, .stSlider label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }

    /* 3. FIX DROPDOWN/INPUT TEXT COLOR (Black text on white boxes) */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* 4. CLINICAL INFO TAB STYLING (The "Different" look) */
    .clinical-card {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid #38bdf8;
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .clinical-header {
        color: #38bdf8 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 15px;
    }
    .clinical-item {
        color: #f1f5f9 !important;
        font-size: 1.2rem !important;
        line-height: 1.6;
        margin-bottom: 10px;
        border-left: 3px solid #38bdf8;
        padding-left: 15px;
    }

    /* 5. TITLES */
    h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= FAST AI ENGINE (CACHED) =================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    # Pre-loading the explainer to save time during diagnosis
    explainer = shap.TreeExplainer(model)
    return model, explainer

try:
    model, explainer = load_assets()
except Exception as e:
    st.error(f"Error loading AI assets: {e}")
    model, explainer = None, None

# ================= HEADER =================
st.markdown("<h1>NeuroGuard Elite</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.2rem; margin-top:-20px;'>Advanced Stroke Risk Assessment System</p>", unsafe_allow_html=True)

tabs = st.tabs(["⚡ Diagnosis Dashboard", "📘 Clinical Intelligence", "🧠 AI Interpretation"])

# ================= TAB 1: DIAGNOSIS =================
with tabs[0]:
    col1, col2 = st.columns([2, 1.2], gap="large")
    
    with col1:
        st.markdown("<h3 style='color:#38bdf8;'>Patient Vitals</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender Selection", ["Male", "Female", "Other"])
        age = c2.slider("Age of Patient", 0, 100, 50)
        
        c3, c4 = st.columns(2)
        glucose = c3.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
        bmi = c4.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
        
        c5, c6 = st.columns(2)
        smoking = c5.selectbox("Smoking History", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        work = c6.selectbox("Employment Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

        c7, c8 = st.columns(2)
        hypertension = c7.radio("Chronic Hypertension?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
        heart_disease = c8.radio("Known Heart Disease?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", horizontal=True)
        
        residence = st.selectbox("Residence Environment", ["Urban", "Rural"])
        married = st.selectbox("Marital Status (Ever Married)", ["Yes", "No"])

    with col2:
        st.markdown("<h3 style='color:#38bdf8; text-align:center;'>Risk Result</h3>", unsafe_allow_html=True)
        predict_btn = st.button("EXECUTE AI DIAGNOSIS", use_container_width=True, type="primary")
        
        if predict_btn and model:
            input_df = pd.DataFrame({
                "gender": [gender], "age": [age], "hypertension": [hypertension],
                "heart_disease": [heart_disease], "ever_married": [married],
                "work_type": [work], "Residence_type": [residence],
                "avg_glucose_level": [glucose], "bmi": [bmi], "smoking_status": [smoking]
            })
            
            # Prediction
            proba = model.predict_proba(input_df)[0][1] * 100
            st.session_state['input_data'] = input_df
            
            # Result Color
            res_color = "#22c55e" if proba < 30 else "#eab308" if proba < 70 else "#ef4444"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba,
                number = {'suffix': "%", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': res_color},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.2)"},
                        {'range': [30, 70], 'color': "rgba(234, 179, 8, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}]
                }
            ))
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"<div style='text-align:center; padding:15px; border-radius:10px; background:{res_color}; color:white; font-weight:900; font-size:1.5rem;'>{ 'CRITICAL RISK' if proba > 70 else 'ELEVATED RISK' if proba > 30 else 'LOW RISK'}</div>", unsafe_allow_html=True)

# ================= TAB 2: CLINICAL INTEL (DIFFERENT LOOK) =================
with tabs[1]:
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.markdown("<div class='clinical-header'>📘 Medical Guidelines & Risks</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='clinical-item'><b>B.E. F.A.S.T Protocol:</b> Essential for early stroke detection. Check Balance, Eyes, Face, Arms, Speech, and Time.</div>", unsafe_allow_html=True)
    st.markdown("<div class='clinical-item'><b>Hypertension:</b> The primary contributor to arterial damage. Values consistently over 140/90 mmHg require clinical review.</div>", unsafe_allow_html=True)
    st.markdown("<div class='clinical-item'><b>Glucose Management:</b> Average glucose levels above 200 mg/dL can double the risk of ischemic stroke.</div>", unsafe_allow_html=True)
    st.markdown("<div class='clinical-item'><b>BMI & Physicality:</b> A BMI > 30 is often associated with obstructive sleep apnea, a hidden stroke risk factor.</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: AI ANALYSIS (SHAP OPTIMIZED) =================
with tabs[2]:
    st.markdown("<h3 style='color:#38bdf8;'>Neural Path Analysis</h3>", unsafe_allow_html=True)
    if 'input_data' in st.session_state:
        with st.spinner("Decoding AI logic..."):
            # SHAP Interpretation
            shap_values = explainer.shap_values(st.session_state['input_data'])
            
            # Adjusting for model output type
            val = shap_values[1] if isinstance(shap_values, list) else shap_values

            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#0f172a')
            
            shap.waterfall_plot(shap.Explanation(
                values=val[0], 
                base_values=explainer.expected_value[1], 
                data=st.session_state['input_data'].iloc[0],
                feature_names=st.session_state['input_data'].columns
            ), show=False)
            
            st.pyplot(fig)
            st.markdown("<p style='text-align:center; color:#94a3b8;'>Red bars show factors increasing risk, Blue bars show factors decreasing it.</p>", unsafe_allow_html=True)
    else:
        st.info("Run a diagnosis in the first tab to unlock this analysis.")
