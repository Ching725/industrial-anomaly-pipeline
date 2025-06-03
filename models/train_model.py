from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from flaml import AutoML
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.base import is_classifier

def train_models(X, y, model_names=None):
    if model_names is None:
        model_names = ["RandomForest", "XGBoost", "LightGBM", "VotingEnsemble", "CatBoost", "FLAML"]

    models = {}
    unique_classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)
    class_weight_dict = dict(zip(unique_classes, class_weights))

    if "RandomForest" in model_names:
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf.fit(X, y)
        models["RandomForest"] = rf

    if "XGBoost" in model_names:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xgb.fit(X, y)
        models["XGBoost"] = xgb

    if "LightGBM" in model_names:
        lgbm = LGBMClassifier(class_weight="balanced", random_state=42)
        lgbm.fit(X, y)
        models["LightGBM"] = lgbm

    if "CatBoost" in model_names:
        cat = CatBoostClassifier(verbose=0, class_weights=class_weight_dict, random_state=42)
        cat.fit(X, y)
        models["CatBoost"] = cat

    if "LogisticRegression" in model_names:
        lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)
        lr.fit(X, y)
        models["LogisticRegression"] = lr

    if "FLAML" in model_names:
        automl = AutoML()
        automl.fit(X, y, task="classification", time_budget=60, verbose=0)
        models["FLAML"] = automl.model
        # Store features used by FLAML for later use
        if hasattr(automl.model, "feature_name_"):
            models["FLAML_selected_features"] = automl.model.feature_name_
        else:
            models["FLAML_selected_features"] = X.columns.tolist()

    if "VotingEnsemble" in model_names:
        base_estimators = [
            (name.lower(), model)
            for name, model in models.items()
            if name not in ("FLAML", "VotingEnsemble")
            and model is not None
            and is_classifier(model)
        ]
        if base_estimators:
            voting = VotingClassifier(estimators=base_estimators, voting='soft')
            voting.fit(X, y)
            models["VotingEnsemble"] = voting

    return models



# # 多模型選擇
# model_choices = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "FLAML", "VotingEnsemble"]
# selected_models = st.multiselect("Select models to train:", model_choices, default=["RandomForest", "XGBoost"])

# if st.button("🚀 Train Selected Models"):
#     trained_models = train_models(X_train, y_train, model_names=selected_models)
#     st.success(f"✅ Trained {len(trained_models)} models: {list(trained_models.keys())}")
