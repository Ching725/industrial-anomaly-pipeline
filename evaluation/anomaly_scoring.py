"""
anomaly_scoring.py

This module provides tools for:
- Computing Isolation Forest anomaly scores
- Integrating SHAP values with anomaly scores (hybrid score)
- Reducing dimensions using PCA
- Generating a report for 'double-flagged' anomalies with top SHAP features
"""


from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import pandas as pd



def compute_isolation_scores(X):
    clf = IsolationForest(contamination=0.05, random_state=42)
    scores = -clf.fit(X).decision_function(X)
    return scores

def reduce_dimensionality(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced

import numpy as np

def compute_hybrid_score(model, X, shap_values, iso_scores, shap_weight=0.4, iso_weight=0.6):
    """
    Combine SHAP mean absolute values and IsolationForest scores into a hybrid anomaly score.
    """
    shap_mean = np.abs(shap_values).mean(axis=1)
    hybrid_scores = shap_weight * shap_mean + iso_weight * iso_scores
    return hybrid_scores, shap_values


def save_double_flagged_report(scores, shap_values, y_true, path="double_flagged_anomalies.csv"):
    threshold = np.percentile(scores, 90)
    double_flagged = (scores >= threshold) & (np.abs(shap_values).mean(axis=1) > 0.1)
    top_features = np.argsort(-np.abs(shap_values), axis=1)[:, :3]
    top3_shap = [[f"f_{idx}" for idx in row] for row in top_features]

    df_report = pd.DataFrame({
        "hybrid_score": scores,
        "double_flagged": double_flagged,
        "true_label": y_true,
        "shap_mean_abs": np.abs(shap_values).mean(axis=1),
        "top3_shap_features": top3_shap
    })
    df_report[double_flagged].to_csv(path, index=False)
    print(f"✅ Saved {path} with SHAP explanations")