import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

from src.data_loader import load_data

df = load_data("data/train.csv")
if df is None:
    exit()
#print("✅ Data loaded successfully")

categorical_features = [ "Gender", "Married", "Education", "Self_Employed", "Property_Area"]
numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
features = categorical_features + numeric_features

#Additional Engineered Features: These are added to improve the accuracy of the model
df["Total_Income"] = df["ApplicantIncome"] + df ["CoapplicantIncome"]
df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"].replace(0, np.nan)   
#This is caculated just for principal and not the total amount. In this code we are not considering interest. 
#Note to self : .replace(0, np.nan) replaces ) with NaN (Not a Number) and not the other way around. here it is done to avoid division by zero. Just like we use param constraints etc in simulink.
df["EMI_to_Income_Ratio"] = df["EMI"] / df["Total_Income"].replace(0, np.nan)
numeric_features += ["Total_Income", "EMI", "EMI_to_Income_Ratio"]
features = categorical_features + numeric_features

target = "Loan_Status"

df.dropna(subset=[target], inplace=True)
#This line of code removes any rows where the target (Loan_Status) is NA and inplace=True updates the same dataframe instead of creating a new one. 
#Even though we are using CatBoostClassifier which can handle missing values, that only makes sense when that is in features.

df[target] = df[target].map({"Y": 1, "N": 0})

df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
df[categorical_features] = df[categorical_features].fillna("Unknown")

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_features_indices = [X.columns.get_loc(col) for col in categorical_features] #Gets index position of each column name in categorical_features. It gives CatBoostClassifier the information of which columns are categorical
model = CatBoostClassifier(
    iterations=1000, #This gives the numer of Trees to build (boosting rounds). more is better but also takes time
    learning_rate=0.05, #Controls how much model corrects errors from the previous iteration. Smaller is slower but more accurate. Typically between 0.01 to 0.1
    depth=6, #Maximum depth of each tree. Deeper is complex but risks overfitting. Typically between 4 adn 10
    cat_features=cat_features_indices, 
    verbose=0, #Controls how much log output CatBoost prints during training. 0 is no log. 1 or higher prints progress bars metrics etc., Eg: 100 to see updates every 100 iterations
    random_state=42 #Sets random seed to ensure consistancy. Useful for debugging across experiments
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ CatBoost Model Trained with {accuracy * 100:.2f}% Accuracy")

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "loan_eligibility_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(model_dir, "columns.pkl"))
joblib.dump(categorical_features, os.path.join(model_dir, "categorical_features.pkl"))