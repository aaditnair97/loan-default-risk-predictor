import shap
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

def explain_model(model_path, X_sample, save_path="shap_summary.png"):
    """
    Generates SHAP summary bar plot for a trained CatBoost model.

    Parameters:
    - model_path: path to .cbm CatBoost model
    - X_sample: pandas DataFrame of test inputs
    - save_path: where to save the plot
    """
    # Load model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Create SHAP explainer and compute values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Plot bar summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ SHAP summary plot saved to {save_path}")
