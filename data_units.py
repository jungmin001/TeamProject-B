import pandas as pd
import numpy


def load_dataset(F):
    print("CSV 파일을 불러와 DataFrame으로 반환합니다.")
    df = pd.read_csv(F)
    return df

def summarize_dataset(df, target_col=None):
    data = {
        "행/열 개수": df.shape,
        "컬럼명": df.columns,
        '결측치 수': df.isna().sum().sum()
    }
    
    if target_col is not None and target_col in df.columns:
        data['타겟 분포'] = df[target_col].value_counts(normalize=True)

    return data



    
    


