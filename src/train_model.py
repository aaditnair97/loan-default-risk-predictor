# train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import joblib

from data_loader import load_data
from evaluate import evaluate_model, bootstrap_auc
from explain import explain_model

# Load data
df = load_data()
X = df.drop(columns=["target"])
y = df["target"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create Pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
test_pool = Pool(X_test, y_test, cat_features=categorical_cols)

# Train CatBoost model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    eval_metric="AUC",
    class_weights=[1, 2],  # Increase weight for minority class
    random_seed=42,
    verbose=0
)
model.fit(train_pool)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print("📊 Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Bootstrap CI
ci_results = bootstrap_auc(model, X_test.to_numpy(), y_test.to_numpy())
print("\n🎯 Bootstrap AUC (95% CI):")
print(f"Mean AUC: {ci_results['bootstrap_auc_mean']:.4f}")
print(f"95% CI: ({ci_results['95% CI'][0]:.4f}, {ci_results['95% CI'][1]:.4f})")

# Save model
os.makedirs("models", exist_ok=True)
model.save_model("models/catboost_model.cbm")
print("\n💾 Model saved to models/catboost_model.cbm")

# SHAP explanation
X_sample = X_test.sample(1000, random_state=42)
explain_model("models/catboost_model.cbm", X_sample)
