import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shap

# def plot_confusion_matrix(y_true, y_pred, label_names, model_name):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=label_names, yticklabels=label_names)
#     plt.title(f"Confusion Matrix - {model_name}")
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.tight_layout()

#     os.makedirs("artifacts/figures", exist_ok=True)
#     plt.savefig(f"artifacts/figures/confusion_matrix_{model_name}.png")
#     plt.close()


def plot_confusion_matrix(y_true, y_pred, label_names, model_name, output_dir="artifacts/figures"):
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    print(y_true)
    print(y_pred)
    print(f"Confusion Matrix:\n{cm}")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return fig  


def save_shap_summary_plot(shap_values, X_df, filename="shap_summary_bar.png"):
    """
    Save a SHAP summary bar plot based on SHAP values and input feature dataframe.

    Parameters:
        shap_values (np.ndarray or list): SHAP values (n_samples x n_features) or list[ndarray] for binary classifiers
        X_df (pd.DataFrame): DataFrame with column names matching SHAP features
        filename (str): Output file path (.png)
    """
    # Handle binary classifier shap output (list of 2 arrays)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # class 1

    shap.summary_plot(shap_values, X_df, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Saved SHAP summary bar plot as {filename}")


import optuna

def save_optuna_visualizations(study, output_dir="artifacts"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 儲存最佳參數
    with open(f"{output_dir}/best_params.json", "w") as f:
        import json
        json.dump(study.best_params, f, indent=4)

    # 儲存最佳分數
    with open(f"{output_dir}/best_score.txt", "w") as f:
        f.write(f"Best Score: {study.best_value}\n")

    # 儲存最佳歷史圖
    ax1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig1 = ax1.figure
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/optuna_optimization_history.png")
    plt.close(fig1)

    # 儲存參數影響力圖
    ax2 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig2 = ax2.figure
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/optuna_param_importance.png")
    plt.close(fig2)

    print(f"✅ 已儲存 Optuna 可視化結果與參數至 {output_dir}/")