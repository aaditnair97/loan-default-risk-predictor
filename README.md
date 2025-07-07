# 💣 Loan Default Risk Predictor

A production-grade machine learning app to predict the **probability of loan default**, trained on real Lending Club data and explained using SHAP.

Built using:
- 🧠 CatBoostClassifier for robust credit scoring
- 📊 SHAP for local & global explainability
- 🌐 Streamlit for interactive web deployment

---

## 🚀 Features

- Trained on 200k+ rows of Lending Club loan data
- Handles class imbalance with `class_weights`
- Evaluates performance with ROC-AUC, PR-AUC, F1 Score
- Bootstraps confidence intervals for model reliability
- Fully explainable predictions using SHAP (force plots)
- Deployed in a Streamlit app with real-time interaction

---

## 📂 Project Structure

```
loan-default-risk-predictor/
├── data/                # (ignored) raw and cleaned Lending Club datasets
├── models/              # (ignored) trained CatBoost model files
├── src/
│   ├── data_loader.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── explain.py
├── streamlit_app/
│   └── app.py           # Streamlit frontend with SHAP visualisation
├── scripts/
│   └── clean_real_data.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance

| Metric      | Value     |
|-------------|-----------|
| ROC-AUC     | **0.7091** |
| PR-AUC      | 0.3573    |
| F1 Score    | 0.3327    |
| Bootstrap AUC CI | (0.7032 – 0.7149) |

---

## 🧠 SHAP Explainability

This model includes full SHAP-based interpretability:

- **Summary plots** to understand global drivers of risk
- **Force plots** to explain individual predictions

---

## 🖥️ Run the App

Install dependencies:

```bash
pip install -r requirements.txt
```

Then launch:

```bash
streamlit run streamlit_app/app.py
```

---

## ⚠️ Limitations

- Trained on US Lending Club data — not directly UK-calibrated
- Only 36 or 60 month loans supported (as per dataset)
- No external macroeconomic or behavioural features included (yet)

---

## 🔮 Future Improvements

- Add UK-specific credit data
- Integrate real-time API input
- Deploy to Streamlit Cloud / Hugging Face Spaces
- Generate PDF credit summaries per user
- Add SHAP waterfall and cohort analysis

---

## 👨‍💻 Author

**Aadit Sabareesh Nair**  
[🔗 LinkedIn](https://www.linkedin.com/in/aadit-sabareesh-nair)


---

## 🌍 Try It Live

👉 **Streamlit App:** [loan-default-risk-predictor]([https://loan-default-risk-predictor.streamlit.app](https://loan-default-risk-predictor-d2keznstgjgadkfxsthkxs.streamlit.app/))

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://loan-default-risk-predictor.streamlit.app](https://loan-default-risk-predictor-d2keznstgjgadkfxsthkxs.streamlit.app/))

---

## 📦 Download Model

Download the trained CatBoost model:  
**[catboost_model.cbm (GitHub Release)](https://github.com/aaditnair97/loan-default-risk-predictor/releases/download/v1.0/catboost_model.cbm)**

---

## 📉 Limitations

- Only supports 36- or 60-month terms due to original dataset structure
- No API or DB integration (can be added)
- Not intended for real-world credit decisions (educational only)

