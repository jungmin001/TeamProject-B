from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def split_xy(df, feature_cols, target_col):
    X = df[feature_cols]
    y = df['failure_soon']
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
    err =  (np.array(y_proba) >= threshold).astype(int)
    return err

def calculate_threshold_cost(y_true, y_pred, cost_fp=10, cost_fn=100):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    
    return total_cost

def find_best_threshold(y_true, y_proba, thresholds, cost_fp=10, cost_fn=100):
    best_threshold = None
    min_cost = float('inf')  
    
    for th in thresholds:
        y_pred = apply_threshold(y_proba, threshold=th)
        current_cost = calculate_threshold_cost(y_true, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)
        
        if current_cost < min_cost:
            min_cost = current_cost
            best_threshold = th
            
    return best_threshold