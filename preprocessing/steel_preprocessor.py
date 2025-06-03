import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocessing.base_preprocessor import apply_feature_preprocessing

def label_encode_fault(df, label_cols=None):
    if label_cols is None:
        label_cols = df.columns[-7:]
    df['fault_type'] = df[label_cols].idxmax(axis=1)
    df = df.drop(columns=label_cols)
    le = LabelEncoder()
    df['fault_type'] = le.fit_transform(df['fault_type'])
    return df, le

# def preprocess_steel(X_train, X_test, options=None, thresholds=None):
#     return apply_feature_preprocessing(X_train, X_test, options, thresholds)