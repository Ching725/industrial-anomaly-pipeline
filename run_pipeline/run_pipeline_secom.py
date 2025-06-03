import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from preprocessing.secom_preprocessor import load_secom_data
from preprocessing.base_preprocessor import apply_feature_preprocessing
from preprocessing.balancer import apply_smoteenn
from models.train_model import train_models
from evaluation.evaluator import evaluate_model, select_best_model
from utils.saver import save_model_bundle, save_hybrid_score_csv
from utils.shap_utils import get_shap_explainer
from evaluation.anomaly_scoring import compute_hybrid_score, save_double_flagged_report
from evaluation.visualizer import save_shap_summary_plot, plot_confusion_matrix
from utils.logger import enable_logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


# === 設定 log 檔輸出 ===
log_dir = "artifacts/secom_logs"
enable_logging(log_dir=log_dir)  # 開啟 log 紀錄
print("這行會同時寫入 terminal + log 檔")
# === 設定 log 檔輸出 ===
log_dir = "artifacts/steel_logs"
enable_logging(log_dir=log_dir)  # 開啟 log 紀錄
print("這行會同時寫入 terminal + log 檔")

def run_pipeline():
    print("🔍 Step 1: Load raw data")
    df, y = load_secom_data("dataset/secom/secom.data", "dataset/secom/secom_labels.data")
    print(f"✅ Raw shape: {df.shape}, labels: {y.value_counts().to_dict()}")

    print("\n✂️ Step 2: Split into train/test")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df, y, stratify=y, test_size=0.2, random_state=42)

    print("\n🧹 Step 3: Preprocess using Preprocessor class")
    options = {
        "drop_missing": True,
        "drop_lowvar": True,
        "drop_corr": True,
        "clip_method": "iqr"  # 或 "iqr"
    }
    thresholds = {
        "drop_missing": 0.2,
        "low_variance": 0.01,
        "high_corr": 0.95,
        "z_thresh": 2.5          # 僅當 clip_method = zscore 時會使用
    }
    X_train_scaled, X_test_scaled, scaler, imputer, removed_features, remaining_columns = apply_feature_preprocessing(X_train_raw, X_test_raw, options=options, thresholds=thresholds)
    print(f"✅ Preprocessed: {X_train_scaled.shape} (train), {X_test_scaled.shape} (test)")
    print(f"Removed features: {removed_features}")
    print(f"X train mean: {X_train_scaled.mean()}")
    print(f"X train std: {X_train_scaled.std()}")
 
    print("\n⚖️ Step 4: Apply SMOTEENN on training set")
    X_train_resampled, y_train_resampled = apply_smoteenn(X_train_scaled, y_train)
    print(f"✅ Resampled: {X_train_resampled.shape}, labels: {np.bincount(y_train_resampled)}")

    print("\n🤖 Step 5: Train VotingClassifier")
    model_names = ["LogisticRegression", "LightGBM", "CatBoost", "VotingEnsemble"]
    models = train_models(X_train_resampled, y_train_resampled, model_names)

    print("\n📊 Step 6: Evaluate model")
    evaluation_results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        plot_confusion_matrix(result["y_true"], result["y_pred"], result["target_names"], "secom_"+name)
        evaluation_results.append(result)

    print("\n⚠️ Step 7: Select best model")
    best_result, score_df = select_best_model(evaluation_results)
    print("\n📊 模型分數比較：")
    print(score_df)
    best_model_name = best_result["model_name"]
    best_model = models[best_model_name]
    print(f"\n🏆 最佳模型是：{best_model_name}")

    print("\n⚠️ Step 8: Compute hybrid anomaly score")
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_scores = -iso.fit(X_train_scaled).decision_function(X_train_scaled)

    explainer = get_shap_explainer(best_model)
    X_train_named = pd.DataFrame(X_train_scaled, columns=remaining_columns)
    
    shap_raw = explainer.shap_values(X_train_named)

    # shap_raw = explainer.shap_values(X_train_named)
    hybrid_scores, shap_values = compute_hybrid_score(best_model, X_train_scaled, shap_raw, iso_scores)
    print("Top 5 hybrid anomaly scores:", hybrid_scores[:5])

    print("\n📊 Step 9: Generate SHAP summary plot")
    save_shap_summary_plot(shap_values, X_train_named)

    print("\n💾 Step 10: Save results and model artifacts")
    save_hybrid_score_csv(hybrid_scores, best_model, iso_scores, shap_values, y_train.values, X_train_scaled, path="hybrid_scores_output.csv")
    preprocessor = {
        "removed_features": removed_features,
        "scaler": scaler,
        "imputer": imputer
    }
    save_model_bundle(model=best_model, preprocessor=preprocessor)


    print("\n📄 Step 11: Export double-flagged anomalies with SHAP interpretation")
    save_double_flagged_report(hybrid_scores, shap_values, y_train.values, path="double_flagged_anomalies.csv")

if __name__ == "__main__":
    run_pipeline()
