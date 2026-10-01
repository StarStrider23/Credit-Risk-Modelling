import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from scipy.stats import spearmanr, pearsonr

from functions import rmse, mae

model_name = "xgb"

val_df = pd.read_csv(f"lgd/{model_name}/{model_name}_validation_df.csv")
test_df = pd.read_csv(f"lgd/{model_name}/{model_name}_test_df.csv")

"""
ACTUAL & PREDICTED PLOTS
"""

# VALIDAITION

Y_val = val_df["lgd_model"]
Y_pred_val = val_df["lgd_pred"]

plt.figure(figsize=(7, 6))

plt.scatter(Y_pred_val, Y_val, alpha=0.6)
plt.plot([0, 1], [0, 1], linestyle="--", color='r')

plt.xlabel("Predicted LGD")
plt.ylabel("Realized LGD")
plt.title(f"Validation Set, Predicted vs Realized, {model_name}")
plt.grid(alpha=0.3)
plt.show()

# TEST

Y_test = test_df["lgd_model"]
Y_pred_test = test_df["lgd_pred"]

plt.figure(figsize=(7, 6))

plt.scatter(Y_pred_test, Y_test, alpha=0.6)
plt.plot([0, 1], [0, 1], linestyle="--", color='r')

plt.xlabel("Predicted LGD")
plt.ylabel("Realized LGD")
plt.title(f"Test Set, Predicted vs Realized, {model_name}")
plt.grid(alpha=0.3)
plt.show()

"""
MAE & RMSE AND CORRELATIONS
"""

spearman_corr, spearman_pval = spearmanr(Y_pred_val, Y_val)
pearson_corr, pearson_pval = pearsonr(Y_pred_val, Y_val)

print("VALIDATION SET")
print("Mean predicted LGD:", Y_pred_val.mean())
print("Mean actual LGD:", Y_val.mean())
print("RMSE: ", rmse(Y_pred_val, Y_val))
print("MAE: ", mae(Y_pred_val, Y_val))
print("Spearman correlation: ", spearman_corr, "p-value: ", spearman_pval)
print("Pearson correlation: ", pearson_corr, "p-value: ", pearson_pval)

spearman_corr, spearman_pval = spearmanr(Y_pred_test, Y_test)
pearson_corr, pearson_pval = pearsonr(Y_pred_test, Y_test)

print("TEST SET")
print("Mean predicted LGD:", Y_pred_test.mean())
print("Mean actual LGD:", Y_test.mean())
print("RMSE: ", rmse(Y_pred_test, Y_test))
print("MAE: ", mae(Y_pred_test, Y_test))
print("Spearman correlation: ", spearman_corr, "p-value: ", spearman_pval)
print("Pearson correlation: ", pearson_corr, "p-value: ", pearson_pval)

"""
RISK GROUPS
"""

# test_results = test_df[["loan_identifier", "default_month", "lgd_model", "lgd_pred"]].copy()

# test_results["risk_group"] = pd.qcut(test_results["lgd_pred"], q=5, duplicates="drop")

# bucket_results = (test_results.groupby("risk_group", observed=True).agg(
#         observations=("lgd_model", "size"),
#         mean_predicted_lgd=("lgd_pred", "mean"),
#         mean_actual_lgd=("lgd_model", "mean"),
#         median_actual_lgd=("lgd_model", "median")
#     ).reset_index()
# )

# print(bucket_results)

"""
ANNUAL PREDICTIONS & TEMPORAL DETERIORATION SHOWCASE
"""

df = pd.concat([val_df, test_df], ignore_index=True)

df["year"] = df["observation_month"] // 100

rows = []

for year in sorted(df["year"].unique()):

    y_model = df["lgd_model"][df["year"] == year]
    y_pred = df["lgd_pred"][df["year"] == year]

    row = {"year": year,
        "average lgd_model": y_model.mean(),
        "average lgd_pred": y_pred.mean(),
        "rmse": rmse(y_model, y_pred),
         "mae": mae(y_model, y_pred)}

    rows.append(row)

month_table = pd.DataFrame(rows)

print(month_table)
