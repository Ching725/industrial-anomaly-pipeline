from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def apply_smoteenn(X, y, random_state=42):
    smoteenn = SMOTEENN(random_state=random_state)
    return smoteenn.fit_resample(X, y)

def apply_smote(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)

def apply_undersample(X, y):
    rus = RandomUnderSampler()
    return rus.fit_resample(X, y)