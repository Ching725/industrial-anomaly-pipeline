"""
shap_utils.py

Provides utility to initialize appropriate SHAP Explainer 
based on model type, including support for VotingClassifier.
"""

import shap
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_shap_explainer(model, fallback_key="lightgbm"):
    # 如果是 VotingEnsemble，抓其中一個模型來解釋
    if isinstance(model, VotingClassifier):
        base_model = model.named_estimators_.get(fallback_key)
        if base_model is None:
            raise ValueError(f"❌ VotingClassifier 中找不到 key='{fallback_key}' 的子模型")
    else:
        base_model = model

    # 根據模型類型使用對應 SHAP Explainer
    if isinstance(base_model, (XGBClassifier, LGBMClassifier, CatBoostClassifier)):
        return shap.TreeExplainer(base_model)
    else:
        return shap.Explainer(base_model)  # fallback for linear or sklearn models