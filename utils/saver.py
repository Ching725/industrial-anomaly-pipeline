

import json
from pathlib import Path
import joblib
import os
import pandas as pd
import numpy as np

import joblib
import json
import os
from pathlib import Path

def save_model_bundle(model, preprocessor, output_dir="artifacts"):
    """
    Save trained model and preprocessing steps to disk.

    Parameters:
        model: trained model
        preprocessor (dict): should include keys 'scaler', 'imputer', 'removed_features', and optionally 'label_encoder'
        output_dir (str): directory to save artifacts
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "trained_model.pkl"))
    joblib.dump(preprocessor["scaler"], os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(preprocessor["imputer"], os.path.join(output_dir, "imputer.pkl"))

    with open(os.path.join(output_dir, "removed_features.json"), "w") as f:
        json.dump(list(preprocessor["removed_features"]), f)

    if "label_encoder" in preprocessor and preprocessor["label_encoder"] is not None:
        joblib.dump(preprocessor["label_encoder"], os.path.join(output_dir, "label_encoder.pkl"))

    print(f"✅ Saved model and preprocessing artifacts to ./{output_dir}/")
    

# preprocessor = {
#     "removed_features": removed_features,
#     "scaler": scaler,
#     "imputer": imputer
# }
# save_artifacts(model=trained_model, preprocessor=preprocessor)


def load_model_bundle(input_dir="artifacts/model_bundle"):
    model = joblib.load(os.path.join(input_dir, "trained_model.pkl"))
    scaler = joblib.load(os.path.join(input_dir, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(input_dir, "label_encoder.pkl"))

    with open(os.path.join(input_dir, "dropped_features.json"), "r") as f:
        dropped_features = json.load(f)

    print(f"✅ 成功載入模型與前處理元件從 {input_dir}/")
    return model, scaler, label_encoder, dropped_features

# model, scaler, label_encoder, dropped_features = load_model_bundle()


def save_hybrid_score_csv(scores, model, iso, shap_values, y_true, X_train_scaled, path="hybrid_scores_output.csv"):
    df = pd.DataFrame({
        "hybrid_score": scores,
        "model_confidence": model.predict_proba(X_train_scaled)[:, 1],
        "iso_score": iso,
        "shap_mean_abs": np.abs(shap_values).mean(axis=1),
        "true_label": y_true
    })
    df.to_csv(path, index=False)
    print(f"✅ Saved {path}")