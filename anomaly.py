import numpy as np


def detect_zscore_anomaly(df, feature_cols, threshold=3.0):
    for col in feature_cols:
       z_score = (df[f'{col}'] - df[f'{col}'].mean()) / df[f'{col}'].std()
       df[f'Z_score_{col}'] = (z_score.abs() >= threshold).astype(int)

    z_cols = []
    for z_col in feature_cols:
        z_cols.append(f'Z_score_{z_col}')
        
    sum = df[z_cols].sum(axis=1)
    df['label'] = np.where(sum != 0,1,0)
    return df['label']