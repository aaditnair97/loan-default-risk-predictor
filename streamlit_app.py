import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests

model = joblib.load("models/loan_eligibility_model.pkl")
columns = joblib.load("models/columns.pkl")

def get_conversion_rates(base="EUR", targets=["USD", "GBP", "INR"]):
    API_KEY = st.secrets["API_KEY"]
    url = f"https://api.exchangeratesapi.io/v1/latest?access_key={API_KEY}&base={base}&symbols={','.join(targets)}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data.get("success"):
                return data["rates"], True
    except Exception:
        pass
    return {"USD": 1.14, "GBP": 0.84, "INR": 97.75}, False

conversion_rates, live_data = get_conversion_rates()
currency = st.selectbox("Select currency", ["GBP", "USD", "EUR", "INR"])

st.markdown("### 💷 Conversion rates (Base: EUR")
for curr, rate in conversion_rates.items():
    st.markdown(f"**1 EUR = {rate} {curr}**")
st.info("✅ Live rates fetched" if live_data else "⚠️ Using fallback conversion rates")

st.title("🏦 Loan Eligibility Predictor")

gender = st.radio("Gender", ["Male", "Female"])
married = st.radio("Married", ["Yes", "No"])
education = st.radio("Education", ["Graduate", "Not Graduate"])
self_employed = st.radio("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

app_income = st.number_input(f"Applicant's Monthly Income (in {currency})", min_value=0)
coapp_income = st.number_input(f"Co-applicant's Monthly Income (in {currency})", min_value=0)
loan_amt = st.number_input(f"Loan Amount (in {currency})", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=1)
credit_history = st.radio("Credit History", [1.0, 0.0], format_func=lambda x: "Yes" if x == 1.0 else "No")

gbp_to_inr = conversion_rates.get("INR", 105.0)
if currency != "INR":
    user_rate = conversion_rates.get(currency, 1.0)
    factor = gbp_to_inr / user_rate
    app_income *= factor
    coapp_income *= factor
    loan_amt *= factor

total_income = app_income + coapp_income
emi = loan_amt / loan_term
emi_ratio = emi / total_income if total_income else 0

sample = {
    "Gender": gender,
    "Married": married,
    "Education": education,
    "Self_Employed": self_employed,
    "Property_Area": property_area,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": coapp_income,
    "LoanAmount": loan_amt,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Total_Income": total_income,
    "EMI": emi,
    "EMI_to_Income_Ratio": emi_ratio
}

sample_df = pd.DataFrame([sample])
sample_df = pd.get_dummies(sample_df)
for col in columns:
    if col not in sample_df.columns:
        sample_df[col] = 0
sample_df = sample_df[columns]

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan Approved ({probability * 100:.2f}% probability)")
    else:
        st.error(f"❌ Loan Not Approved ({probability * 100:.2f}% probability)")

    st.markdown("---")
    st.caption("⚠️ This tool is an educational ML project. Please consult financial institutions for real decisions.")