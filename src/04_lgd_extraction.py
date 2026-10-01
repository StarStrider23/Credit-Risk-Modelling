import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg:///" "credit_risk")

lgd_history = pd.read_sql(
    """
    SELECT * 
    FROM lgd.lgd_history
    """,
    engine
)

common_cols = ['loan_identifier', 'default_month']
feature_cols = ['observation_month', 'observation_type', 'current_actual_upb', 'current_loan_delinquency_status',
                'loan_age', 'current_interest_rate', 'eltv', 'modification_flag', 'payment_deferral_flag']
target_cols = ['zero_balance_code', 'zero_balance_effective_date', 'delinquent_accrued_interest', 
               'zero_balance_removal_upb', 'actual_loss']

t1 = lgd_history[lgd_history["observation_type"] == "1_month_before"].copy()
t1 = t1[common_cols + feature_cols]
t1 = t1.reset_index(drop=True)

t = lgd_history[lgd_history["observation_type"] == "default_month"].copy()
t = t[common_cols + target_cols]
t = t.reset_index(drop=True)

valid_lgd = (t["delinquent_accrued_interest"].notna() & t["actual_loss"].notna() & t["zero_balance_removal_upb"].notna())

t_clean = t[valid_lgd].copy()

t_clean["ead"] = (t_clean["delinquent_accrued_interest"] + t_clean["zero_balance_removal_upb"])
t_clean["lgd"] = (t_clean["actual_loss"] / t_clean["ead"])

t1_clean = t1.merge(t_clean[common_cols], on=common_cols, how="inner")

lgd = t1_clean.merge(t_clean, on=common_cols, how="inner")

analysis = lgd.copy()

analysis["modification_flag"] = analysis["modification_flag"].fillna("None")
analysis["payment_deferral_flag"] = analysis["payment_deferral_flag"].fillna("None")

lgd_origination = pd.read_sql(
    """
    SELECT * 
    FROM lgd.origination_data
    """,
    engine
)

lgd_full = analysis.merge(lgd_origination, on=["loan_identifier"], how="inner")

lgd_full.drop(columns=["observation_type"], inplace=True)
lgd_full["lgd_model"] = lgd_full["lgd"].clip(0, 1)

import seaborn as sb
import matplotlib.pyplot as plt

cols = [    
    "eltv",
    "ltv",
    "cltv",
    "current_actual_upb",
    "loan_age",
    "credit_score",
    "dti",
    "interest_rate",
    "current_interest_rate",
    "lgd_model"
    ]

plt.figure(figsize=(12,8))
sb.heatmap(lgd_full[cols][lgd_full["observation_month"] <= 202112].corr(numeric_only=True, method="spearman"))
plt.show()

print(lgd_full[cols][lgd_full["observation_month"] <= 202112].corr(numeric_only=True, method="spearman"))


# lgd_full.to_parquet("data/lgd_full.csv") 