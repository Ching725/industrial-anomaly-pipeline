import pandas as pd
from preprocessing.base_preprocessor import apply_feature_preprocessing
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_secom_data(data_path, label_path):
    df = pd.read_csv(data_path, sep=' ', header=None)
    labels = pd.read_csv(label_path, sep=' ', header=None)
    y = (labels.iloc[:, 0] == 1).astype(int)
    return df, y

# def preprocess_secom(X_train, X_test, options=None, thresholds=None):
#     return apply_feature_preprocessing(X_train, X_test, options, thresholds)


class Preprocessor:
    def __init__(self, correlation_threshold=0.95, variance_threshold=1e-5):
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self.selected_columns = None
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X: pd.DataFrame):
        # Step 1: Impute
        print("▶ Imputing missing values with mean...")
        X_imputed = self.imputer.fit_transform(X)
        X_df = pd.DataFrame(X_imputed, columns=X.columns)

        # Step 2: IQR-based outlier clip
        print("▶ Handling outliers using IQR clipping...")
        Q1 = X_df.quantile(0.25)
        Q3 = X_df.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - 1.5 * IQR
        self.upper_bound = Q3 + 1.5 * IQR
        X_clipped = X_df.clip(lower=self.lower_bound, upper=self.upper_bound, axis=1)

        # Step 3: Remove low variance
        print("▶ Removing near-zero variance features...")
        stds = X_clipped.std()
        X_var = X_clipped.loc[:, stds > self.variance_threshold]
        print(f"✅ Remaining features after variance filter: {X.shape[1]}")

        # Step 4: Remove highly correlated features
        print(f"▶ Removing highly correlated features (Pearson > {self.correlation_threshold})...")
        corr_matrix = X_var.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        X_selected = X_var.drop(columns=to_drop)

        print(f"✅ Remaining features after correlation filter: {X.shape[1]}")

        print("▶ Scaling features...")
        self.selected_columns = X_selected.columns.tolist()

        # Step 5: Fit scaler
        self.scaler.fit(X_selected)

        return self

    def transform(self, X: pd.DataFrame):
        X_imputed = self.imputer.transform(X)
        X_df = pd.DataFrame(X_imputed, columns=X.columns)
        X_clipped = X_df.clip(lower=self.lower_bound, upper=self.upper_bound, axis=1)
        X_selected = X_clipped[self.selected_columns]
        X_scaled = self.scaler.transform(X_selected)
        return X_scaled

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)
