import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def apply_feature_preprocessing(X_train, X_test, options=None, thresholds=None):
    if options is None:
        options = {
            "drop_missing": True,
            "drop_lowvar": True,
            "drop_corr": True,
            "clip_method": "None"
        }
    if thresholds is None:
        thresholds = {
            "drop_missing": 0.4,
            "low_variance": 1e-4,
            "high_corr": 0.95,
            "z_thresh": 3
        }

    removed_features = set()

    # 1️⃣ 移除缺失比例過高欄位
    if options.get("drop_missing", False):
        thresh = thresholds["drop_missing"]
        keep_cols = X_train.columns[X_train.isnull().mean() < thresh]
        removed_missing = set(X_train.columns) - set(keep_cols)
        removed_features.update(removed_missing)
        X_train = X_train[keep_cols]
        X_test = X_test[keep_cols]

    # 2️⃣ 移除低變異欄位
    if options.get("drop_lowvar", False):
        selector = VarianceThreshold(threshold=thresholds["low_variance"])
        before_cols = X_train.columns
        X_train = pd.DataFrame(selector.fit_transform(X_train), columns=np.array(before_cols)[selector.get_support()])
        removed_lowvar = set(before_cols) - set(X_train.columns)
        removed_features.update(removed_lowvar)
        X_test = X_test[X_train.columns]

    # 3️⃣ 移除高度相關欄位
    if options.get("drop_corr", False):
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > thresholds["high_corr"])]
        removed_corr = set(to_drop)
        removed_features.update(removed_corr)
        X_train = X_train.drop(columns=to_drop, errors="ignore")
        X_test = X_test.drop(columns=to_drop, errors="ignore")

    # 4️⃣ Clip outliers
    clip_method = options.get("clip_method", None)
    if clip_method == "iqr":
        Q1 = X_train.quantile(0.25)
        Q3 = X_train.quantile(0.75)
        IQR = Q3 - Q1
        X_train = X_train.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
        X_test = X_test.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
    elif clip_method == "zscore":
        z_thresh = thresholds.get("z_thresh", 3)
        std_bounds = get_std_bounds(X_train, X_train.columns, z_thresh)
        X_train = apply_std_clip(X_train, std_bounds)
        X_test = apply_std_clip(X_test, std_bounds)

    # 5️⃣ 缺失值補全
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)

    # 6️⃣ 標準化
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    # 7️⃣ 回傳剩餘欄位順序
    remaining_columns = X_train.columns.tolist()

    return X_train, X_test, scaler, imputer, removed_features, remaining_columns


# 補充用：標準差裁剪邏輯
def get_std_bounds(df, columns, z_thresh=3):
    bounds = {}
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - z_thresh * std
        upper = mean + z_thresh * std
        bounds[col] = (lower, upper)
    return bounds

def apply_std_clip(df, bounds):
    df_clipped = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df_clipped[col] = df[col].clip(lower, upper)
    return df_clipped



# options = {
#     "drop_missing": True,
#     "drop_lowvar": True,
#     "drop_corr": True,
#     "clip_method": "zscore"  # 或 "iqr"
# }
# thresholds = {
#     "drop_missing": 0.2,
#     "low_variance": 0.01,
#     "high_corr": 0.95,
#     "z_thresh": 2.5          # 僅當 clip_method = zscore 時會使用
# }


from sklearn.preprocessing import LabelEncoder

def encode_labels(y, task_type="binary"):
    """
    Encode target labels for classification tasks.

    Parameters:
        y (array-like): target labels
        task_type (str): "binary" or "multiclass"

    Returns:
        y_encoded: encoded labels
        label_encoder: fitted LabelEncoder object or None
    """
    if task_type == "multiclass":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded, le
    else:
        return y, None