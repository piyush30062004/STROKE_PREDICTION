import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="NeuroGuard Elite | Stroke AI",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🧠"
)

# ================= ULTRA-VISIBILITY UI CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Poppins:wght@400;700;900&display=swap');

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        background-attachment: fixed;
    }

    /* GLOBAL TEXT - MAX CONTRAST */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
    }
    
    label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        margin-bottom: 10px !important;
    }

    .css-card {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
    }

    h1 {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 4.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h3 {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* DROPDOWN & INPUT FIXES */
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 2px solid #475569 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
    }

    li[data-baseweb="option"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* CLINICAL TAB SPECIFIC VISIBILITY FIX */
    .clinical-text {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        line-height: 1.8 !important;
        text-shadow: 0px 0px 8px rgba(255,255,255,0.2);
    }

    .stButton > button {
        width: 100%;
        border-radius: 15px !important;
        height: 4.5em;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
    }

    div[data-testid="stSlider"] {
        background-color: #0f172a !important;
        border: 2px solid #475569 !important;
        padding: 15px 25px 10px 25px;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= HELPERS =================
def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop": continue
        if transformer == "passthrough": feature_names.extend(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            try: feature_names.extend(transformer.get_feature_names_out(cols))
            except: feature_names.extend(transformer.get_feature_names_out())
        else: feature_names.extend(cols)
    return feature_names

# ================= LOAD MODEL =================
try:
    model = joblib.load("model.pkl")
except:
    model = None

# ================= HEADER =================
st.markdown("<h1>NeuroGuard Elite</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.3rem; color: #94a3b8; font-weight:600; margin-top:-30px; margin-left:8px;'>Advanced Clinical Decision Support System</p>", unsafe_allow_html=True)

tabs = st.tabs(["⚡ Prediction Dashboard", "📘 Clinical Info", "🧠 AI Analysis", "📈 Metrics", "📄 Export Report"])

# ================= PREDICTION TAB =================
with tabs[0]:
    col_input, col_action = st.columns([3, 1.5], gap="large")
    with col_input:
        st.markdown('<div class="css-card"><h3>👤 Patient Demographics</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: gender = st.selectbox("Gender Selection", ["Male", "Female", "Other"])
        with c2: age = st.slider("Current Age", 0, 100, 45)
        with c3: residence = st.selectbox("Residence Environment", ["Urban", "Rural"])
        c4, c5 = st.columns(2)
        with c4: work_type = st.selectbox("Employment Sector", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        with c5: ever_married = st.selectbox("Marital Status", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="css-card"><h3>🏥 Clinical Vitals</h3>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: glucose = st.number_input("Average Glucose Level (mg/dL)", 40.0, 300.0, 85.0)
        with m2: bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 24.5)
        with m3: smoking = st.selectbox("Current Smoking History", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        m4, m5 = st.columns(2)
        with m4: hypertension = st.radio("Chronic Hypertension", [0, 1], format_func=lambda x: "Present" if x==1 else "Absent", horizontal=True)
        with m5: heart_disease = st.radio("Known Heart Disease", [0, 1], format_func=lambda x: "Present" if x==1 else "Absent", horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_action:
        st.markdown('<div style="margin-top:25px;"></div>', unsafe_allow_html=True)
        predict_btn = st.button("RUN AI DIAGNOSIS")
        
        if predict_btn and model:
            input_data = pd.DataFrame({"gender":[gender],"ever_married":[ever_married],"work_type":[work_type],"Residence_type":[residence],"smoking_status":[smoking],"age":[age],"hypertension":[hypertension],"heart_disease":[heart_disease],"avg_glucose_level":[glucose],"bmi":[bmi]})
            proba = model.predict_proba(input_data)[0][1] * 100
            st.session_state.input_data, st.session_state.proba = input_data, proba
            
            color = "#10b981" if proba < 25 else "#f59e0b" if proba < 65 else "#ef4444"
            status = "LOW RISK" if proba < 25 else "ELEVATED RISK" if proba < 65 else "CRITICAL RISK"
            
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=proba, number={'suffix': "%", 'font': {'color': color, 'size': 55, 'weight': 'bold'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "#ffffff"}, 'bar': {'color': color}, 'bgcolor': "rgba(0,0,0,0)",
                'steps': [{'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.2)'},{'range': [25, 65], 'color': 'rgba(245, 158, 11, 0.2)'},{'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.2)'}]}))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ffffff"}, height=350)
            
            st.markdown(f'<div class="css-card" style="border: 3px solid {color}; text-align: center; background: rgba(15, 23, 42, 0.9);">', unsafe_allow_html=True)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f'<div style="background:{color}; color:white; padding:12px 40px; border-radius:50px; display:inline-block; font-weight:900; font-size:1.4rem;">{status}</div></div>', unsafe_allow_html=True)

# ================= CLINICAL INFO TAB (FIXED TEXT VISIBILITY) =================
with tabs[1]:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("## 📘 Clinical Insights")
    
    st.markdown("""
    <div class="clinical-text">
    A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients.
    </div>
    """, unsafe_allow_html=True)
    
    c_inf1, c_inf2 = st.columns(2)
    with c_inf1:
        st.markdown("### 🔍 Risk Indicators")
        st.markdown("""
        <div class="clinical-text">
        • <b>Hypertension:</b> The primary driver of vascular damage and artery hardening.<br>
        • <b>Glucose:</b> High levels indicate potential diabetic complications affecting blood flow.<br>
        • <b>BMI:</b> High Body Mass Index correlates with increased arterial pressure and metabolic stress.
        </div>
        """, unsafe_allow_html=True)
    with c_inf2:
        st.markdown("### ⚠️ Emergency Signs")
        st.markdown("""
        <div class="clinical-text">
        • <b>F.A.S.T:</b> Face drooping, Arm weakness, Speech difficulty, Time to call emergency.<br>
        • Sudden numbness or weakness in the face, arm, or leg, especially on one side of the body.<br>
        • Sudden confusion, trouble speaking, or difficulty understanding speech.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ================= AI ANALYSIS TAB =================
with tabs[2]:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("## 🧠 Neural Interpretability Analysis")
    if "input_data" not in st.session_state:
        st.info("💡 Run a diagnosis to unlock AI analysis.")
    elif model:
        try:
            preprocessor = model.named_steps["preprocess"]
            classifier = model.named_steps["classifier"]
            X_trans = preprocessor.transform(st.session_state.input_data)
            feat_names = get_feature_names(preprocessor)
            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(X_trans)
            
            if isinstance(shap_vals, list):
                sv, ev = shap_vals[1][0], explainer.expected_value[1]
            else:
                sv = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0]
                ev = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value

            fig, ax = plt.subplots(figsize=(12, 6))
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#1e293b')
            shap.waterfall_plot(shap.Explanation(values=sv, base_values=ev, data=X_trans[0], feature_names=feat_names), show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Analysis Engine Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ================= METRICS & EXPORT =================
with tabs[3]:
    st.markdown('<div class="css-card"><h2>📈 Patient Metrics</h2>', unsafe_allow_html=True)
    if "input_data" in st.session_state:
        fig_bar = px.bar(x=["Glucose", "BMI", "Age"], y=[glucose, bmi, age], template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="css-card"><h2>📄 Export Summary</h2><p>Click below to generate a clinical PDF summary.</p>', unsafe_allow_html=True)
    if st.button("GENERATE REPORT"): st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)
