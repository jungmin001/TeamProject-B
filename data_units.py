import pandas as pd
import numpy


def load_dataset(F):
    try:
        print("LogTemp : Warning File load Complete! :D")
        return pd.read_csv(F)
    except FileNotFoundError:   # File path error
        print("LogTemp : Error Failed to load CSV file!! ;(")
        return None
    except Exception as e:      # No file
        print("LogTemp : Error Failed to load CSV file!! ;(")
        return None

def summarize_dataset(df, target_col=None):
    data = {
        "행/열 개수": df.shape,
        "컬럼명": df.columns,
        '결측치 수': df.isna().sum().sum()
    }
    
    if target_col is not None and target_col in df.columns:
        data['타겟 분포'] = df[target_col].value_counts(normalize=True)

    return data



    
    


