
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path="data/Churn_Modelling.csv"):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    df = df.copy()
    for c in ['RowNumber','CustomerId','Surname']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    return df

def add_engineered_features(df):
    df = df.copy()
    if 'Age' in df.columns and 'Tenure' in df.columns:
        df['TenureByAge'] = df['Tenure'] / (df['Age'] + 1)
    if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
        df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    if 'CreditScore' in df.columns and 'Age' in df.columns:
        df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + 1)
    return df

def encode_and_scale(df, scaler=None, fit_scaler=True):
    """
    Label-encode categorical columns and scale numeric features.
    Returns: (df_transformed, label_encoders_map, scaler)
    """
    df = df.copy()
    le_map = {}
    for c in ['Geography','Gender','Card Type']:
        if c in df.columns:
            df[c] = df[c].astype(str)
            le = LabelEncoder()
            if fit_scaler:
                df[c] = le.fit_transform(df[c])
            else:
                df[c] = le.transform(df[c])
            le_map[c] = le
    if 'EstimatedSalary' in df.columns:
        df['EstimatedSalary'] = df['EstimatedSalary'].fillna(df['EstimatedSalary'].median())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'Exited']
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, le_map, scaler

def split_xy(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def balance_with_smote(X_train, y_train, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res
