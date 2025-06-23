import pandas as pd

print("📥 Reading raw Lending Club CSV...")
df = pd.read_csv(
    "data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
    low_memory=False
)

print("🧹 Cleaning data...")
df = df[df.loan_status.isin(["Fully Paid", "Charged Off"])].copy()
df["target"] = (df["loan_status"] == "Charged Off").astype(int)

cols = [
    "loan_amnt", "term", "int_rate", "grade", "emp_length", "annual_inc",
    "dti", "fico_range_low", "inq_last_6mths", "open_acc", "pub_rec", "target"
]
df = df[cols].dropna()

# Clean text fields
df["term"] = df["term"].str.extract(r"(\d+)").astype(float)
df["emp_length"] = df["emp_length"].str.extract(r"(\d+)").fillna(0).astype(int)

# Sample
df = df.sample(n=200000, random_state=42)

print("💾 Saving cleaned dataset to data/lending_club_clean.csv")
df.to_csv("data/lending_club_clean.csv", index=False)
print("✅ Done!")
