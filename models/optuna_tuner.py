import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np

def tune_lightgbm_with_optuna(X, y, n_trials=30):
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 31, 256),
            'max_depth': trial.suggest_int("max_depth", 4, 15),
            'min_child_samples': trial.suggest_int("min_child_samples", 5, 50),
            'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 20),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
            'min_gain_to_split': 0.0,
            'verbosity': -1,
            'random_state': 42
        }
        model = lgb.LGBMClassifier(**params)
        score = cross_val_score(model, X, y, cv=3, scoring='f1_weighted').mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["objective"] = "multiclass"
    best_params["num_class"] = len(np.unique(y))
    return lgb.LGBMClassifier(**best_params), study
