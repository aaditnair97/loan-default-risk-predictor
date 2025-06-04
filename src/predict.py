import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

model_path = "models/loan_eligibility_model.pkl"
columns_path = "models/columns.pkl"

model = joblib.load(model_path)
columns = joblib.load(columns_path)

sample_data = {
    "Gender": "Male",
    "Married": "Yes",
    "Education": "Graduate",
    "Self_Employed": "No",
    "Property_Area": "Urban",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0
}

sample_df = pd.DataFrame([sample_data])
sample_df["Total_Income"] = sample_df["ApplicantIncome"] + sample_df["CoapplicantIncome"]
sample_df["EMI"] = sample_df["LoanAmount"] / sample_df["Loan_Amount_Term"].replace(0, np.nan)
sample_df["EMI_to_Income_Ratio"] = sample_df["EMI"] / sample_df["Total_Income"].replace(0, np.nan)

for col in columns:
    if col not in sample_df.columns:
        sample_df[col] = 0
sample_df = sample_df[columns]  # reorder columns

categorical_features = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

prediction = model.predict(sample_df)[0]
probability = model.predict_proba(sample_df)[0][1]

label = "✅ Approved" if prediction == 1 else "❌ Not Approved"
print(f"💡 Prediction: {label} ({probability * 100:.2f}% probability)")