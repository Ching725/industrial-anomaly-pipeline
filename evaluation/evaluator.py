"""
evaluator.py

Tools for:
- Evaluating classification models
- Plotting confusion matrices
- Summarizing and selecting best model results across experiments
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import numpy as np

def predict_labels(model, X, task_type="multiclass", threshold=0.5):
    """
    Unified prediction function supporting binary and multiclass classification.

    Parameters:
        model: trained model
        X: input features (e.g., X_test or new data)
        task_type (str): "binary" or "multiclass"
        threshold (float): classification threshold for binary

    Returns:
        y_pred: predicted class labels
        y_prob: predicted probabilities (or None if unavailable)
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)

        if task_type == "binary":
            if y_prob.shape[1] == 2:
                y_pred = (y_prob[:, 1] >= threshold).astype(int)
            else:
                y_pred = (y_prob[:, 0] >= threshold).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

        return y_pred, y_prob

    else:
        y_pred = model.predict(X)
        return y_pred, None
    
#  y_pred, y_prob = predict_labels(model, X_test_input, task_type=task_type)  # "binary" or "multiclass"


# def evaluate_model(model, X_test, y_test, label_encoder=None, model_name="model", task_type="multiclass"):
#     y_pred, _ = predict_labels(model, X_test, task_type=task_type)

#     y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 and y_test.shape[1] > 1 else y_test

#     if label_encoder:
#         target_names = label_encoder.classes_.astype(str)
#     else:
#         target_names = [str(c) for c in sorted(np.unique(y_true))]

#     report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
#     return {
#         "model_name": model_name,
#         "y_true": y_true,
#         "y_pred": y_pred,
#         "report": report,
#         "target_names": target_names
#     }

def evaluate_model(model, X_test, y_test, label_encoder=None, model_name="model", task_type="multiclass"):
    y_pred, _ = predict_labels(model, X_test, task_type=task_type)

    # 是否需要 inverse_transform
    if label_encoder:
        y_true = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)
        target_names = label_encoder.classes_.astype(str)
    else:
        # y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 and y_test.shape[1] > 1 else y_test
        y_true = np.array(y_test.sort_index().tolist())
        target_names = [c for c in sorted(np.unique(y_true))]
        print("target_names", target_names)
        # target_names = ["Pass", "False"]

    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return {
        "model_name": model_name,
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report,
        "target_names": target_names
    }
# def evaluate_model(model, X_test, y_test, label_encoder=None, model_name="model"):
#     if hasattr(model, "predict_proba"):
#         y_pred_probs = model.predict_proba(X_test)
#         y_pred = np.argmax(y_pred_probs, axis=1)
#     else:
#         y_pred = model.predict(X_test)

#     if y_test.ndim > 1 and y_test.shape[1] > 1:
#         y_true = np.argmax(y_test, axis=1)
#     else:
#         y_true = y_test

#     if label_encoder:
#         target_names = label_encoder.classes_.astype(str)
#     else:
#         target_names = [str(c) for c in sorted(np.unique(y_true))]

#     report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
#     return {
#         "model_name": model_name,
#         "y_true": y_true,
#         "y_pred": y_pred,
#         "report": report,
#         "target_names": target_names
#     }



def summarize_model_scores(evaluation_results):
    summary = []
    for result in evaluation_results:
        report = result["report"]
        summary.append({
            "Model": result["model_name"],
            "Accuracy": report["accuracy"],
            "F1 (macro)": np.mean([report[label]["f1-score"] for label in report if label not in ["accuracy", "macro avg", "weighted avg"]]),
            "F1 (weighted)": report["weighted avg"]["f1-score"]
        })
    return pd.DataFrame(summary).sort_values(by="F1 (weighted)", ascending=False)


def select_best_model(evaluation_results):
    scores_df = summarize_model_scores(evaluation_results)
    best_model_name = scores_df.iloc[0]["Model"]
    best_result = next(res for res in evaluation_results if res["model_name"] == best_model_name)
    return best_result, scores_df



