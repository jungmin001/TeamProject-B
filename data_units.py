import pandas as pd
import numpy

def load_dataset(F):
    try:
        print("LogTemp : Warning File load Complete! :D")
        return pd.read_csv(F)
    except FileNotFoundError:   # File path error
        print("LogTemp : Error Failed to load CSV file!! ;(")
        return None
    except Exception:            # No file
        print("LogTemp : Error Failed to load CSV file!! ;(")
        return None

def summarize_dataset(df,target_col=None):
    df_summ = pd.DataFrame({"행/열 개수" : df.shape,
                            "컬럼명" : df.columns,
                            '결측치 수' : df.isna().sum(),
                            '타깃 분포' : df[f'{target_col}'].value_counts(normalize=True)})
    return df_summ



    
    


