import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import numpy as np
import streamlit.components.v1 as components
import uuid
import os

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Loan Default Risk", layout="centered")

# Load trained CatBoost model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("models/catboost_model.cbm")
    return model

model = load_model()

# App title and intro
st.title("💣 Loan Default Risk Predictor")
st.markdown("""
Enter loan applicant details to predict the **probability of default**.  
This model is trained on real Lending Club data and is explained with SHAP.
""")

# --- Input form ---
loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=50000, value=15000, step=500)
term = st.selectbox("Loan Term (months)", options=[36, 60])
int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=13.0, step=0.1)
grade = st.selectbox("Credit Grade", options=list("ABCDEFG"))
emp_length = st.slider("Employment Length (years)", 0, 10, 3)
annual_inc = st.number_input("Annual Income", min_value=10000, max_value=500000, value=75000, step=1000)
dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 18.0, step=0.1)
fico_range_low = st.slider("FICO Score (Lower Bound)", 600, 850, 690, step=10)
inq_last_6mths = st.slider("Credit Inquiries (6 months)", 0, 10, 1)
open_acc = st.slider("Open Credit Accounts", 0, 30, 8)
pub_rec = st.slider("Public Records", 0, 5, 0)

# --- Construct input DataFrame ---
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

# --- Predict and explain ---
if st.button("🚀 Predict Default Risk"):
    prob_default = model.predict_proba(input_df)[0][1]
    st.metric("💣 Probability of Default", f"{prob_default:.2%}")

    # SHAP explanation
    st.subheader("🔍 SHAP Force Plot Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Save SHAP plot to temp HTML and embed
    shap_id = str(uuid.uuid4())
    tmp_html = f"shap_force_{shap_id}.html"
    shap.save_html(tmp_html, shap.force_plot(
        explainer.expected_value, shap_values, input_df
    ))

    with open(tmp_html, "r", encoding="utf-8") as f:
        components.html(f.read(), height=400, scrolling=True)

    os.remove(tmp_html)
