import pandas as pd
import numpy

def load_dataset(F):
    print("CSV 파일을 불러와 DataFrame으로 반환합니다.")
    df = pd.read_csv(F)
    return df

def summarize_dataset(df,target_col=None):
    df_summ = pd.DataFrame({"행/열 개수":df.shape,
                        "컬럼명": df.columns,
                        '결측치 수':df.isna().sum(),
                        '타깃 분포':df[f'{target_col}'].value_counts(normalize=True)})
    return df_summ



    
    


