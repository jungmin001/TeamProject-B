import numpy as np

def detect_zscore_anomaly(df, feature_cols, threshold=3.0):
    for col in feature_cols:
        z_score = (df[col] - df[col].mean()) / df[col].std()
        df[f'Z_score_{col}'] = (z_score.abs() >= threshold).astype(int)
        
    z_cols = [f'Z_score_{col}' for col in feature_cols]
    
    return df[z_cols].max(axis=1).sum()