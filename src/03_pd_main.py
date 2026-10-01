import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.metrics import roc_curve, precision_recall_curve

model_name = "xgb"

val_df = pd.read_csv(f"pd/{model_name}/{model_name}_validation_df.csv")
test_df = pd.read_csv(f"pd/{model_name}/{model_name}_test_df.csv")

Y_val = val_df["default_12m"]
Y_val_prob = val_df["pd_12m"]

Y_test = test_df["default_12m"]
Y_test_prob = test_df["pd_12m"]

"""
ROC AUC & PR AUC
"""

# VALIDAITION SET

roc_auc_v = roc_auc_score(Y_val, Y_val_prob)
pr_auc_v = average_precision_score(Y_val, Y_val_prob)
brier_v = brier_score_loss(Y_val, Y_val_prob)
logloss_v = log_loss(Y_val, Y_val_prob)

print("VALIDATION SET: ")
print("ROC-AUC: ", roc_auc_v)
print("PR-AUC: ", pr_auc_v)
print("Brier score: ", brier_v)
print("Log Loss: ", logloss_v)
print("Average Predicted Default Rate: ", Y_val_prob.mean())
print("Actual default rate: ", Y_val.mean())

# TEST SET

roc_auc_t = roc_auc_score(Y_test, Y_test_prob)
pr_auc_t = average_precision_score(Y_test, Y_test_prob)
brier_t = brier_score_loss(Y_test, Y_test_prob)
logloss_t = log_loss(Y_test, Y_test_prob)

print("\n")

print("TEST SET: ")
print("ROC-AUC: ", roc_auc_t)
print("PR-AUC: ", pr_auc_t)
print("Brier score: ", brier_t)
print("Log Loss: ", logloss_t)
print("Average Predicted Default Rate: ", Y_test_prob.mean())
print("Actual default rate: ", Y_test.mean()) 

"""
ROC AUC & PR AUC CURVES
"""

# VALIDATION SET

fpr_v, tpr_v, _ = roc_curve(Y_val, Y_val_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr_v, tpr_v, color="b", label=f"{model_name} (ROC AUC = {roc_auc_v:.3f})")
plt.plot([0, 1], [0, 1], color="r", linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Validation set, ROC AUC Curve, {model_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


precision_v, recall_v, _ = precision_recall_curve(Y_val, Y_val_prob)
baseline_v = Y_val.mean()

plt.figure(figsize=(7, 6))
plt.plot(recall_v, precision_v, color="b", label=f"{model_name} (PR AUC = {pr_auc_v:.3f})")
plt.axhline(baseline_v, color="r", linestyle="--", label=f"Random baseline = {baseline_v:.4%}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Validation set, PR AUC Curve, {model_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# TEST SET

fpr_t, tpr_t, _ = roc_curve(Y_test, Y_test_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr_t, tpr_t, color="b", label=f"{model_name} (ROC AUC = {roc_auc_t:.3f})")
plt.plot([0, 1], [0, 1], color="r", linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Test set, ROC AUC Curve, {model_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


precision_t, recall_t, _ = precision_recall_curve(Y_test, Y_test_prob)
baseline_t = Y_test.mean()

plt.figure(figsize=(7, 6))
plt.plot(recall_t, precision_t, color="b", label=f"{model_name} (PR AUC = {pr_auc_t:.3f})")
plt.axhline(baseline_t, color="r", linestyle="--", label=f"Random baseline = {baseline_t:.4%}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Test set, PR AUC Curve, {model_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

"""
MONTHLY PREDICTIONS & TEMPORAL DETERIORATION SHOWCASE
"""

df = pd.concat([val_df, test_df], ignore_index=True)

rows = []

for month in sorted(df["observation_month"].unique()):

    y = df["default_12m"][df["observation_month"] == month]
    p = df["pd_12m"][df["observation_month"] == month]

    row = {"observation_month": month,
        "defaults": y.sum(),
        "observations": len(y),
        "avg_predicted_probability": p.mean(),
         "actual_default_rate": y.mean()}

    if y.nunique() == 2:
        row["roc_auc"] = roc_auc_score(y, p)
        row["pr_auc"] = average_precision_score(y, p)
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan

    rows.append(row)

month_table = pd.DataFrame(rows)

print(month_table)

"""
RANKING
"""

test_ranking = pd.DataFrame({"predicted": Y_test_prob, "actual": Y_test})
test_ranking = test_ranking.sort_values("predicted", ascending=False)

total_defaults = test_ranking["actual"].sum()
total_observations = len(test_ranking)

for percentage in [1, 5, 10, 20, 30, 50]:
    
    n = int(total_observations * percentage / 100)

    top_group = test_ranking.iloc[:n]

    defaults_captured = top_group["actual"].sum()

    capture_rate = defaults_captured / total_defaults

    group_default_rate = top_group["actual"].mean()

    lift = group_default_rate / test_ranking["actual"].mean()

    print(f"Top {percentage}%: "
        f"{defaults_captured:.0f} / {total_defaults:.0f} defaults "
        f"({capture_rate:.1%}), "
        f"Lift: {lift:.2f}x"
    )
