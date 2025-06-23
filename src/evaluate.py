# evaluate.py

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.utils import resample
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Computes evaluation metrics for a classification model.
    Returns ROC-AUC, PR-AUC, F1-Score.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_probs)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_preds)

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "F1-score": f1
    }

def bootstrap_auc(model, X_test, y_test, n_iterations=1000, random_state=42):
    """
    Computes bootstrap confidence intervals for ROC-AUC.
    Returns mean AUC and 95% confidence interval.
    """
    rng = np.random.RandomState(random_state)
    scores = []

    for _ in range(n_iterations):
        indices = rng.randint(0, len(X_test), len(X_test))
        X_sample = X_test[indices]
        y_sample = y_test[indices]

        y_prob_sample = model.predict_proba(X_sample)[:, 1]
        auc_score = roc_auc_score(y_sample, y_prob_sample)
        scores.append(auc_score)

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    mean_score = np.mean(scores)

    return {
        "bootstrap_auc_mean": mean_score,
        "95% CI": (lower, upper)
    }
