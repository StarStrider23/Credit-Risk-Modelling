import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

model_name = "xgb"

val_df = pd.read_csv(f"pd/{model_name}/{model_name}_validation_df.csv")
test_df = pd.read_csv(f"pd/{model_name}/{model_name}_test_df.csv")

Y_val = val_df["default_12m"].values
Y_val_prob = val_df["pd_12m"].values

Y_test = test_df["default_12m"].values
Y_test_prob = test_df["pd_12m"].values

"""
CALIBRATION
"""

roc_auc = roc_auc_score(Y_test, Y_test_prob)
pr_auc = average_precision_score(Y_test, Y_test_prob)
brier = brier_score_loss(Y_test, Y_test_prob)
logloss = log_loss(Y_test, Y_test_prob)

print("Before calibration: ")

print("ROC-AUC: ", roc_auc)
print("PR-AUC: ", pr_auc)
print("Brier score: ", brier)
print("Log Loss: ", logloss)
print("Actual default rate: ", Y_test.mean())

print("Min:", Y_test_prob.min())
print("Max:", Y_test_prob.max())
print("Mean:", Y_test_prob.mean())

eps = 1e-15

Y_val_prob_clipped = np.clip(Y_val_prob.astype(np.float64), eps, 1-eps)

Y_val_logit = np.log(Y_val_prob_clipped / (1 - Y_val_prob_clipped))

calibrator = LogisticRegression(C=1e6, solver="lbfgs")
calibrator.fit(Y_val_logit.reshape(-1, 1), Y_val)

Y_test_prob_clipped = np.clip(Y_test_prob.astype(np.float64), eps, 1-eps)

Y_test_logit = np.log(Y_test_prob_clipped / (1 - Y_test_prob_clipped))

Y_test_prob_cal = calibrator.predict_proba(Y_test_logit.reshape(-1,1))[:, 1]

roc_auc_cal = roc_auc_score(Y_test, Y_test_prob_cal)
pr_auc_cal = average_precision_score(Y_test, Y_test_prob_cal)
brier_cal = brier_score_loss(Y_test, Y_test_prob_cal)
logloss_cal = log_loss(Y_test, Y_test_prob_cal)

print("After calibration: ")

print("ROC-AUC: ", roc_auc_cal)
print("PR-AUC: ", pr_auc_cal)
print("Brier score: ", brier_cal)
print("Log Loss: ", logloss_cal)
print("Actual default rate: ", Y_test.mean())

print("Min:", Y_test_prob_cal.min())
print("Max:", Y_test_prob_cal.max())
print("Mean:", Y_test_prob_cal.mean())

"""
CALIBRATION CURVE
"""

calibration_df = pd.DataFrame({"predicted" : Y_test_prob, "actual" : Y_test})
calibration_df["risk_group"] = pd.qcut(calibration_df["predicted"], q=10, duplicates="drop")

calibration_table = calibration_df.groupby("risk_group", observed=True).agg(observations=("actual", "size"),
        avg_predicted_probability=("predicted", "mean"),
        actual_default_rate=("actual", "mean"),
        defaults=("actual", "sum"))

plt.figure(figsize=(12,8))

plt.plot(calibration_table["actual_default_rate"], calibration_table["avg_predicted_probability"], 
         color="b", label="Calibration Curve")
plt.plot(calibration_table["actual_default_rate"], calibration_table["actual_default_rate"], 
         color="r", label="Reference Curve")

plt.xlabel("Actual Default Rate")
plt.ylabel("Average Predcited Probability")
plt.title(f"Calibration Curve, {model_name}")

plt.legend()
plt.grid(True, alpha=0.2, color = "grey", ls='--', lw = 1,)
plt.show()

"""
MONTHLY PREDICTIONS & TEMPORAL DETERIORATION SHOWCASE
"""

rows = []

df = test_df.copy()
df["pd_12m_cal"] = Y_test_prob_cal

for month in sorted(df["observation_month"].unique()):

    y = df["default_12m"][df["observation_month"] == month]
    p = df["pd_12m"][df["observation_month"] == month]
    p_cal = df["pd_12m_cal"][df["observation_month"] == month]

    row = {"observation_month": month,
        "defaults": y.sum(),
        "observations": len(y),
        "avg_predicted_probability": p.mean(),
        "avg_predicted_probability_cal": p_cal.mean(),
         "actual_default_rate": y.mean()}

    rows.append(row)

month_table = pd.DataFrame(rows)

print(month_table)