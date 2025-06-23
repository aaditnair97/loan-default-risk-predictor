import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import numpy as np
import streamlit.components.v1 as components
import uuid
import os
import urllib.request

# ✅ Set Streamlit page config
st.set_page_config(page_title="Loan Default Risk", layout="centered")

MODEL_URL = "https://sandbox.openai.com/attachments/streamlit_catboost_model.cbm"
MODEL_PATH = "models/catboost_model.cbm"

# --- Download model if not already available ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        st.info("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# --- UI Header ---
st.title("💣 Loan Default Risk Predictor")
st.markdown("Enter applicant details below to predict the **probability of loan default**.")

# --- Input Widgets ---
loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=50000, value=15000, step=500)
term = st.selectbox("Loan Term (months)", options=[36, 60])
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0, 0.1)
grade = st.selectbox("Credit Grade", list("ABCDEFG"))
emp_length = st.slider("Employment Length (years)", 0, 10, 3)
annual_inc = st.number_input("Annual Income", 10000, 500000, 75000, 1000)
dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 18.0, 0.1)
fico_range_low = st.slider("FICO Score (Lower Bound)", 600, 850, 690, 10)
inq_last_6mths = st.slider("Credit Inquiries (6 months)", 0, 10, 1)
open_acc = st.slider("Open Credit Accounts", 0, 30, 8)
pub_rec = st.slider("Public Records", 0, 5, 0)

input_df = pd.DataFrame([{
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "grade": grade,
    "emp_length": emp_length,
    "annual_inc": annual_inc,
    "dti": dti,
    "fico_range_low": fico_range_low,
    "inq_last_6mths": inq_last_6mths,
    "open_acc": open_acc,
    "pub_rec": pub_rec
}])

# --- Predict and Explain ---
if st.button("🚀 Predict Default Risk"):
    prob_default = model.predict_proba(input_df)[0][1]
    st.metric("💣 Probability of Default", f"{prob_default:.2%}")

    st.subheader("🔍 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    shap_id = str(uuid.uuid4())
    tmp_html = f"shap_force_{shap_id}.html"
    shap.save_html(tmp_html, shap.force_plot(explainer.expected_value, shap_values, input_df))

    with open(tmp_html, "r", encoding="utf-8") as f:
        components.html(f.read(), height=400, scrolling=True)

    os.remove(tmp_html)
