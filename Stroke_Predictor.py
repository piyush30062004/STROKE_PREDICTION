import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(page_title="NeuroGuard Elite", layout="wide", initial_sidebar_state="collapsed")

# ================= DARK MODE VISIBILITY CSS =================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        background-attachment: fixed;
    }

    /* Force all text labels to be pure white and visible */
    label, .stMarkdown p, .stSelectbox label, .stSlider label {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }

    /* Clinical & Plan Card Styling */
    .clinical-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .clinical-header {
        color: #38bdf8 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 20px;
        border-bottom: 1px solid #38bdf8;
    }
    .plan-item {
        background: rgba(56, 189, 248, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #38bdf8;
        color: #e2e8f0 !important;
    }
    .critical-item {
        background: rgba(239, 68, 68, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ef4444;
        color: #fca5a5 !important;
    }

    h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_model()

# ================= HEADER =================
st.markdown("<h1>NeuroGuard Elite</h1>", unsafe_allow_html=True)

tabs = st.tabs(["⚡ Diagnosis Dashboard", "📘 Clinical Intelligence", "🛡️ Personalized Prevention Plan"])

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
            
            # Predict Probability
            proba = model.predict_proba(input_df)[0][1] * 100
            st.session_state['input_data'] = input_df
            st.session_state['proba'] = proba
            
            res_color = "#22c55e" if proba < 30 else "#eab308" if proba < 70 else "#ef4444"
            risk_label = 'CRITICAL RISK' if proba > 70 else 'ELEVATED RISK' if proba > 30 else 'LOW RISK'
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = proba,
                number = {'suffix': "%", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': res_color},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.1)"},
                        {'range': [30, 70], 'color': "rgba(234, 179, 8, 0.1)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.1)"}]
                }
            ))
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"<div style='text-align:center; padding:15px; border-radius:10px; background:{res_color}; color:white; font-weight:900; font-size:1.5rem;'>{risk_label}</div>", unsafe_allow_html=True)

            # --- DOWNLOAD BUTTON ---
            st.markdown("<br>", unsafe_allow_html=True)
            report_text = f"NeuroGuard Elite Report\nAge: {age}\nRisk Score: {proba:.2f}%\nDiagnosis: {risk_label}\nBMI: {bmi}\nGlucose: {glucose}"
            st.download_button("📥 Download Report", report_text, f"Stroke_Report_{age}.txt", use_container_width=True)

# ================= TAB 2: CLINICAL INTEL =================
with tabs[1]:
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.markdown("<div class='clinical-header'>Clinical Intelligence & Risk Protocols</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem;'><b>B.E. F.A.S.T Protocol:</b> Balance, Eyes, Face, Arms, Speech, Time.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem;'><b>Hypertension:</b> Primary contributor to arterial damage. Target < 130/80 mmHg.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem;'><b>Glucose:</b> Levels > 140 mg/dL lead to arterial inflammation.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: UNIQUE FEATURE - PREVENTION PLAN =================
with tabs[2]:
    st.markdown("<h3 style='color:#38bdf8;'>🛡️ Personalized Prevention Roadmap</h3>", unsafe_allow_html=True)
    
    if 'input_data' in st.session_state:
        p_data = st.session_state['input_data'].iloc[0]
        p_risk = st.session_state['proba']
        
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        
        # 1. Immediate Risk-Based Action
        if p_risk > 50:
            st.markdown("<div class='critical-item'>⚠️ <b>HIGH RISK DETECTED:</b> Schedule a carotid ultrasound and consultation with a neurologist within 7 days.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='plan-item'>✅ <b>MAINTENANCE MODE:</b> Continue routine screenings every 6 months.</div>", unsafe_allow_html=True)

        # 2. Dynamic Lifestyle Prescriptions
        st.markdown("<h4 style='color:white;'>Targeted Interventions:</h4>", unsafe_allow_html=True)
        
        if p_data['avg_glucose_level'] > 120:
            st.markdown("<div class='plan-item'>🩸 <b>Glycemic Control:</b> Your glucose is elevated. Reduce refined sugar intake and monitor A1C levels.</div>", unsafe_allow_html=True)
        
        if p_data['bmi'] > 28:
            # FIX: Corrected typo 'unsafe_allow_യിrue' to 'unsafe_allow_html=True'
            st.markdown("<div class='plan-item'>⚖️ <b>Weight Management:</b> Reducing BMI by just 5% can lower stroke risk by nearly 20%. Consider a low-sodium Mediterranean diet.</div>", unsafe_allow_html=True)
            
        if p_data['smoking_status'] in ['smokes', 'formerly smoked']:
            st.markdown("<div class='plan-item'>🚭 <b>Arterial Health:</b> Nicotine thickens blood. Complete cessation is the single most effective way to prevent clot formation.</div>", unsafe_allow_html=True)
        
        if p_data['hypertension'] == 1:
            st.markdown("<div class='plan-item'>❤️ <b>Pressure Monitoring:</b> Since you have hypertension, daily logging of BP is mandatory. Limit salt to < 1,500mg/day.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Please run a diagnosis first to generate your custom prevention plan.")
