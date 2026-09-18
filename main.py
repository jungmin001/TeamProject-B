import data_units
import anomaly
import model_units

print("MIMII 데이터")
df = data_units.load_dataset("data/28_mimii_features_sample_102_260917_3.csv")

summ = data_units.summarize_dataset(df)
print(summ)
feature_cols = ["rms", "spectral_centroid", "zero_crossing_rate"]

err_count = anomaly.detect_zscore_anomaly(df, feature_cols, threshold=3.0)
print("이상 탐지:",err_count)

print("CMAPSS 데이터")

df_c = data_units.load_dataset("data/28_cmapss_fd001_sample_102_260917_2.csv")
summ_c = data_units.summarize_dataset(df_c, "failure_soon")
print(summ_c)


feature_cols = ["sensor_2","sensor_3","sensor_4","sensor_7","sensor_11","sensor_15"]

X,y = model_units.split_xy(df_c, feature_cols, df_c['failure_soon'])


X_train , X_test , y_train , y_test = model_units.split_train_test(X, y, test_size=0.2, random_state=42)

model =model_units.train_random_forest(X_train, y_train, random_state=42)

y_pred = model.predict(X_test)
accuracy , tn, fp,fn,tp= model_units.evaluate_model(y_test, y_pred)
print(f"정확도: {accuracy:.4f} | TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

probability = model_units.predict_failure_probability(model, X_test)
y_proba = probability

err = model_units.apply_threshold(y_proba, threshold=0.5)

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

total_cost= model_units.calculate_threshold_cost(y_test, y_pred, cost_fp=10, cost_fn=100)
print(f"기본 모델 총 비용: {total_cost}")

best_threshold=  model_units.find_best_threshold(y_test, y_proba, thresholds, cost_fp=10, cost_fn=100)
print(f"최적 임계값: {best_threshold}")
