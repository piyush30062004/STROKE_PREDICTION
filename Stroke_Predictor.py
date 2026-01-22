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

    /* Clinical Intelligence Styling */
    .clinical-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .clinical-header {
        color: #38bdf8 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 20px;
        border-bottom: 1px solid #38bdf8;
    }
    .clinical-section {
        margin-bottom: 25px;
        padding-left: 15px;
        border-left: 3px solid #818cf8;
    }
    .section-title {
        color: #818cf8 !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        text-transform: uppercase;
    }
    .section-content {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        line-height: 1.7;
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

# ================= FIXED AI ENGINE (COMPATIBLE WITH PIPELINES) =================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    # Using the generic Explainer to handle Pipeline objects automatically
    explainer = shap.Explainer(model.predict, shap.maskers.Independent(data=np.zeros((1, 10))))
    return model, explainer

try:
    model, explainer = load_assets()
except Exception as e:
    st.error(f"AI Engine Syncing Error: {e}")
    model, explainer = None, None

# ================= HEADER =================
st.markdown("<h1>NeuroGuard Elite</h1>", unsafe_allow_html=True)

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
            
            proba = model.predict_proba(input_df)[0][1] * 100
            st.session_state['input_data'] = input_df
            
            res_color = "#22c55e" if proba < 30 else "#eab308" if proba < 70 else "#ef4444"
            
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
            
            st.markdown(f"<div style='text-align:center; padding:15px; border-radius:10px; background:{res_color}; color:white; font-weight:900; font-size:1.5rem;'>{ 'CRITICAL RISK' if proba > 70 else 'ELEVATED RISK' if proba > 30 else 'LOW RISK'}</div>", unsafe_allow_html=True)

# ================= TAB 2: CLINICAL INTEL (DETAILED) =================
with tabs[1]:
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.markdown("<div class='clinical-header'>Clinical Intelligence & Risk Protocols</div>", unsafe_allow_html=True)
    
    # Section 1
    st.markdown("""<div class='clinical-section'>
        <div class='section-title'>1. Acute Stroke Identification (B.E. F.A.S.T.)</div>
        <div class='section-content'>
            Stroke is a time-critical medical emergency. Use the BE-FAST protocol:<br>
            • <b>Balance:</b> Watch for sudden loss of coordination.<br>
            • <b>Eyes:</b> Check for sudden blurred or double vision.<br>
            • <b>Face:</b> Ask the patient to smile; look for facial drooping.<br>
            • <b>Arms:</b> Ask them to raise both arms; check if one drifts downward.<br>
            • <b>Speech:</b> Listen for slurring or difficulty repeating simple phrases.<br>
            • <b>Time:</b> If any signs are present, call emergency services immediately.
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Section 2
    st.markdown("""<div class='clinical-section'>
        <div class='section-title'>2. Hypertension & Vascular Resistance</div>
        <div class='section-content'>
            Chronic high blood pressure is the primary contributor to ischemic strokes. 
            Hypertension weakens the arterial walls over time, leading to either blockage 
            (clot) or rupture (hemorrhage). Standard clinical target is below 130/80 mmHg.
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Section 3
    st.markdown("""<div class='clinical-section'>
        <div class='section-title'>3. Metabolic Biomarkers (Glucose & BMI)</div>
        <div class='section-content'>
            • <b>Glucose:</b> Patients with Diabetes are 1.5 times more likely to have a stroke. 
            Hyperglycemia causes inflammation and arterial plaque buildup.<br>
            • <b>BMI:</b> High Body Mass Index is often correlated with Sleep Apnea and High 
            Cholesterol, both of which are secondary drivers of cerebrovascular disease.
        </div>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: AI ANALYSIS (COMPATIBLE SHAP) =================
with tabs[2]:
    st.markdown("<h3 style='color:#38bdf8;'>AI Feature Contribution</h3>", unsafe_allow_html=True)
    if 'input_data' in st.session_state:
        with st.spinner("Analyzing neural pathways..."):
            try:
                # Optimized SHAP for Pipelines
                shap_values = explainer(st.session_state['input_data'])
                
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.style.use('dark_background')
                fig.patch.set_facecolor('#0f172a')
                
                shap.plots.bar(shap_values[0], show=False)
                st.pyplot(fig)
                st.markdown("<p style='text-align:center; color:#94a3b8;'>Features with longer bars had the highest influence on this specific patient's risk score.</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"SHAP Visualization Error: {e}")
                st.info("The AI interpretation requires the input data to match the model's training format exactly.")
    else:
        st.info("Complete a Diagnosis Dashboard session first to see the AI breakdown.")
