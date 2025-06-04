# 🏦 Loan Eligibility Predictor

A Machine Learning-based web app to predict whether a user is eligible for a loan, built with Streamlit and CatBoostClassifier.  
This tool takes various personal and financial details as input and provides a probability-based eligibility prediction.

> ⚠️ **Disclaimer:** This project is for educational purposes only and should not be considered as real financial advice.

---

## 🚀 Live App

👉 [Try the Live Demo on Streamlit](https://loan-eligibility-predictor-1.streamlit.app/)

---

## 📌 Features

- Built using a real Indian banking dataset from a Kaggle loan prediction challenge
- Accepts inputs in **GBP, USD, EUR, or INR** and automatically converts to INR
- Predicts **loan approval probability**
- Displays **live or fallback conversion rates**
- Trained with advanced model (CatBoost) + engineered features (EMI ratio, total income etc.)
- User-friendly Streamlit interface

---

## 📊 Inputs Considered

| Feature              | Description                                      |
|----------------------|--------------------------------------------------|
| Gender               | Male or Female                                   |
| Marital Status       | Married or Not                                   |
| Education            | Graduate or Not                                  |
| Self Employment      | Yes or No                                        |
| Property Area        | Urban / Rural / Semiurban                        |
| Applicant Income     | Monthly income of applicant                      |
| Co-applicant Income  | Monthly income of co-applicant                   |
| Loan Amount          | Total requested loan amount                      |
| Loan Term            | Repayment period in months                       |
| Credit History       | Whether credit history exists (1.0 = Yes, 0.0 = No) |

---

## 🧠 ML Pipeline

### ✅ Models Tried

| Model                  | Accuracy | Notes |
|------------------------|----------|-------|
| Logistic Regression    | ~78%     | Weak on non-linear patterns |
| Random Forest          | ~80%     | Better but relied too heavily on Loan Term |
| **CatBoostClassifier** | ✅ ~80%   | Best balance of accuracy and interpretability |

### 🔨 Feature Engineering

- `Total_Income = ApplicantIncome + CoapplicantIncome`
- `EMI = LoanAmount / Loan_Amount_Term`
- `EMI_to_Income_Ratio = EMI / Total_Income`

These features drastically improved the model’s understanding of financial capacity.

---

## 💷 Currency Conversion

The app supports multi-currency input (INR, GBP, USD, EUR) using [ExchangeRatesAPI.io](https://exchangeratesapi.io/).  
- Base currency for conversion: **EUR**  
- Base currency for model prediction: **INR**

If API limits are exceeded, fallback rates are used.

---

## 🚫 Limitations

- The original training data is from India and was in **INR**.
- The model was not trained to understand **cost of living, inflation, or risk standards** in other countries.
- Currency conversion allows usability in other regions, but predictions may be skewed.
- We discovered that the model sometimes gives unrealistic approvals or rejections when loan amount and term ratios deviate from typical values.
- This limitation was diagnosed by testing extreme combinations and observing patterns in the model’s logic.

---

## 🛠 Improvements Made

- Replaced Logistic Regression → Random Forest → CatBoostClassifier
- Re-engineered features to better represent applicant financials
- Replaced manual dummy encoding + scaling with native CatBoost handling
- Implemented live currency conversion with fallback
- Handled missing columns with intelligent imputation for prediction time
- Designed clean Streamlit UI

---

## 🔍 Future Work

- Integrate SHAP explainability for model decisions
- Train with more diverse global datasets
- Add income-to-loan approval insights
- Improve UX and error handling on frontend

---

## 👨‍💻 Author

Made with ❤️ by Aadit Sabareesh Nair  
[www.aaditnair.com](https://www.aaditnair.com)

---

## 📁 Project Structure

