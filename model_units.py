from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def split_xy(df, feature_cols, target_col):
    X = df[feature_cols]
    y = df['label']
    return X,y

def split_train_test(X, y, test_size=0.2, random_state=42):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
    return X_train , X_test , y_train , y_test


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train,y_train)
    return model

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    tn,fp ,fn,tp = confusion_matrix(y_test, y_pred).ravel()
    return accuracy , tn, fp,fn,tp


def predict_failure_probability(model, X_test):
    probability = model.predict_proba(X_test)[:, 1]
    return probability

def apply_threshold(y_proba, threshold=0.5):
#     """예측 확률을 임계값 기준으로 0 또는 1로 변환합니다."""
    

def calculate_threshold_cost(y_true, y_pred, cost_fp=10, cost_fn=100):
#     """FP와 FN 비용을 사용해 총비용을 계산합니다."""


def find_best_threshold(y_true, y_proba, thresholds, cost_fp=10, cost_fn=100):
#     """후보 임계값별 총비용을 비교하고 비용이 가장 낮은 임계값을 반환합니다."""


# `find_best_threshold()`는 후보 임계값별 결과표와 최적 임계값을 함께 반환하도록 작성합니다. 반환 형식은 `dict`, `tuple`, `DataFrame` 중 팀에서 정해도 됩니다. 단, `main.py` 또는 `main.ipynb`에서 후보 임계값별 총비용과 최적 임계값을 출력할 수 있어야 합니다.