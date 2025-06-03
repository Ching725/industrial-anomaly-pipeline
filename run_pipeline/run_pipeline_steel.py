
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import sys
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

from preprocessing.steel_preprocessor import label_encode_fault
from preprocessing.base_preprocessor import apply_feature_preprocessing
from preprocessing.balancer import apply_smoteenn
from models.train_model import train_models
from models.optuna_tuner import tune_lightgbm_with_optuna
from evaluation.evaluator import evaluate_model, summarize_model_scores, select_best_model
from evaluation.visualizer import plot_confusion_matrix
from utils.shap_utils import get_shap_explainer
from evaluation.visualizer import save_shap_summary_plot, save_optuna_visualizations
from utils.saver import save_model_bundle
from utils.logger import enable_logging


# from src.preprocessing import preprocess_features, balance_data
# from src.steel.model_comparator import train_multiple_models, evaluate_models
# from src.steel.optuna_tuner import tune_lightgbm_with_optuna
# from src.steel.optuna_visualizer import save_optuna_visualizations
# from src.steel.model_saver import save_model_bundle
# from src.steel.shap_explainer import explain_model_with_shap

# === 設定 log 檔輸出 ===
log_dir = "artifacts/steel_logs"
enable_logging(log_dir=log_dir)  # 開啟 log 紀錄
print("這行會同時寫入 terminal + log 檔")

def run_pipeline():
    print("🔍 Step 1: Load raw data")
    df = pd.read_csv("dataset/steel/steel_faults.csv")

    print("\n Step 2: Label encode fault_type")
    df, le = label_encode_fault(df)
    X = df.drop(columns=["fault_type"])
    y = df["fault_type"]

    print("\n✂️ Step 3: Split into train/test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("\n🧹 Step 4: Preprocess using Preprocessor class")
    X_train_scaled, X_test_scaled, scaler, imputer, removed_features, remaining_columns = apply_feature_preprocessing(X_train, X_test, options=None, thresholds=None)
    print(f"✅ Preprocessed: {X_train_scaled.shape} (train), {X_test_scaled.shape} (test)")

    print("\n🧹 Step 4.5: Describe stat summary after preprocessing")
    summary = X_train_scaled.describe().T
    print(summary.sort_values("std"))

    os.makedirs("artifacts/figures", exist_ok=True)

    print("\n⚖️ Step 5: Apply SMOTEENN on training set")
    X_train_resampled, y_train_resampled = apply_smoteenn(X_train_scaled, y_train)

    print("\n🤖 Step 6: Muti-Models Training and VotingClassifier")
    model_names = ["RandomForestClassifier", "XGBClassifier", "LightGBM", "CatBoost", "FAML", "VotingEnsemble"]
    models = train_models(X_train_resampled, y_train_resampled, model_names)
    # result_df, best_model, _, _ = evaluate_models(models, X_test_scaled, y_test, le)

    print("\n📊 Step 7: Evaluate model")
    evaluation_results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        plot_confusion_matrix(result["y_true"], result["y_pred"], result["target_names"], "secom_"+name)
        evaluation_results.append(result)

    print("\n⚠️ Step 8: Select best model")
    best_result, score_df = select_best_model(evaluation_results)
    print("\n📊 模型分數比較：")
    print(score_df)
    best_model_name = best_result["model_name"]
    best_model = models[best_model_name]
    print(f"\n🏆 最佳模型是：{best_model_name}")

    print("\🔧 Step 9: Optuna-tuned LightGBM")
    lgbm_opt, study = tune_lightgbm_with_optuna(X_train_resampled, y_train_resampled, n_trials=30)
    lgbm_opt.fit(X_train_resampled, y_train_resampled)
    
    print("📊 Step 10: Save Optuna parameter visualize result")
    save_optuna_visualizations(study, output_dir="artifacts/optuna")

    print("Step 11: Evaluate Optuna-tuned LGBM model")
    Optuna_evaluation_result = evaluate_model(lgbm_opt, X_test_scaled, y_test, label_encoder=le, model_name="Optuna_LGBM")
    Optuna_score_df = summarize_model_scores([Optuna_evaluation_result])
    
    # 比較兩者 recall 分數後選擇最終儲存模型
    print("📊 Step 12: Compare recall scores")
    y_pred_best = best_model.predict(X_test_scaled)
    y_pred_optuna = lgbm_opt.predict(X_test_scaled)
    recall_best = recall_score(y_test, y_pred_best, average='macro')
    recall_optuna = recall_score(y_test, y_pred_optuna, average='macro')
    print(f"📈 Ensemble/AutoML macro Recall: {recall_best:.4f}")
    print(f"📈 Optuna-tuned LGBM macro Recall: {recall_optuna:.4f}")

    final_model = lgbm_opt if recall_optuna >= recall_best else best_model
    final_model_name = "Optuna_LGBM" if recall_optuna >= recall_best else "Ensemble_or_AutoML_Best"
    print(f"✅ 儲存最終模型：{final_model_name}")


    print("📊 Step 13: Save SHAP summary plot")
    explainer = get_shap_explainer(final_model)
    shap_values = explainer(X_train_resampled)
    save_shap_summary_plot(shap_values, X_train_resampled, filename="artifacts/shap_outputs/steel_shap_summary_bar.png")

    print("💾 Step 14: Save results and model artifacts")
    preprocessor = {
        "removed_features": removed_features,
        "scaler": scaler,
        "imputer": imputer,
        "label_encoder": le
    }
    save_model_bundle(final_model, preprocessor, output_dir="artifacts/steel_shap_summary_bar.png")


    print("✅ 完成！模型比較結果如下：")
    print("📊 Compare Macro Recall / AUC / PR AUC of all models")
    # 🧾 合併兩組模型比較結果
    combined_results = pd.concat([score_df, Optuna_score_df], ignore_index=True)

    print(combined_results.sort_values("F1 (macro)", ascending=False))


if __name__ == "__main__":
    run_pipeline()
