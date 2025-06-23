#!/bin/bash
set -e

mkdir -p data

echo "📥 Downloading dataset..."
curl -L -o data/loan_data_full.csv "https://ndownloader.figshare.com/files/22121477"

echo "🧼 Cleaning & sampling..."
"/c/Users/aadit/AppData/Local/Programs/Python/Python313/python.exe" - << 'PYCODE'
import pandas as pd

# Load full dataset
df = pd.read_csv("data/loan_data_full.csv", low_memory=False)

# Keep only two target classes
df = df[df.loan_status.isin(["Fully Paid", "Charged Off"])]

# Create binary target
df["target"] = (df["loan_status"] == "Charged Off").astype(int)

# Select usable columns
cols = [
    "loan_amnt", "term", "int_rate", "grade", "emp_length", "annual_inc",
    "dti", "fico_range_low", "inq_last_6mths", "open_acc", "pub_rec", "target"
]
df = df[cols]

# Clean text fields
df["term"] = df["term"].str.extract(r"(\d+)").astype(float)
df["emp_length"] = df["emp_length"].str.extract(r"(\d+)").fillna(0).astype(int)

# Downsample for faster training
df = df.sample(n=200000, random_state=42)

# Save cleaned CSV
df.to_csv("data/lending_club_clean.csv", index=False)
print("✅ Saved to data/lending_club_clean.csv")
PYCODE
