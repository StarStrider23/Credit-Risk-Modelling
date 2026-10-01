import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
from functions import *
from sklearn.linear_model import LogisticRegression

engine = create_engine("postgresql+psycopg:///" "credit_risk")

"""
PD
"""

pd_df = pd.read_sql(
    f"""
    SELECT *
    FROM ecl.ecl
    """,
    engine,
)

preprocessor_pd = joblib.load("pd/preprocessor.joblib")
model_pd = joblib.load("pd/xgb/xgb.joblib")

pd_df_preprocessed = preprocessor_pd.transform(pd_df.drop(columns=["loan_identifier", "observation_month", "zero_balance_code"]))

pd_df["pd_12m"] = model_pd.predict_proba(pd_df_preprocessed)[:, 1]

"""
CALIBRATION
"""

df_cal = pd.read_csv("pd/xgb/xgb_test_df_15-24.csv")
Y_prob = df_cal["pd_12m"].values
Y = df_cal["default_12m"].values

eps = 1e-15

Y_prob_clipped = np.clip(Y_prob.astype(np.float64), eps, 1-eps)

Y_logit = np.log(Y_prob_clipped / (1 - Y_prob_clipped))

calibrator = LogisticRegression(C=1e6, solver="lbfgs")
calibrator.fit(Y_logit.reshape(-1, 1), Y)

Y_test_prob_clipped = np.clip(pd_df["pd_12m"].values.astype(np.float64), eps, 1-eps)

Y_test_logit = np.log(Y_test_prob_clipped / (1 - Y_test_prob_clipped))

pd_df["pd_12m_cal"] = calibrator.predict_proba(Y_test_logit.reshape(-1,1))[:, 1]

"""
LGD
"""

cols = [
    "loan_identifier",
    "observation_month",
    "eltv",
    "ltv",
    "cltv",
    "current_actual_upb",
    "loan_age",
    "credit_score",
    "dti",
    "current_loan_delinquency_status",
    "property_state",
    "zero_balance_code",
    "interest_rate",
    "current_interest_rate"
]

columns = ", ".join(cols)

lgd_df = pd.read_sql(
    f"""
    SELECT {columns}
    FROM ecl.ecl
    """,
    engine,
)

# STAGE ---------

lgd_df["stage"] = 1

lgd_df["delinquency_bucket"] = lgd_df["current_loan_delinquency_status"].apply(delinquency_bucket)

lgd_df.loc[lgd_df["delinquency_bucket"].isin(["30-59", "60-89", "90-119", "120+"]), "stage"] = 2

lgd_df.loc[lgd_df["zero_balance_code"].isin([2, 3, 9]), "stage"] = 3

# ---------------

preprocessor_lgd = joblib.load("lgd/preprocessor.joblib")
model_lgd = joblib.load("lgd/xgb/xgb.joblib")

lgd_df_preprocessed = preprocessor_lgd.transform(lgd_df.drop(columns=["loan_identifier", "observation_month", "zero_balance_code", "stage"]))
lgd_predictions = model_lgd.predict(lgd_df_preprocessed)

lgd_df["lgd"] = lgd_predictions.clip(0, 1)

"""
EAD
"""

lgd_df["ead"] = lgd_df["current_actual_upb"]

"""
ECL
"""

d = ["eltv",
    "ltv",
    "cltv",
    "current_actual_upb",
    "loan_age",
    "credit_score",
    "dti"]

df = (pd_df[["loan_identifier", "observation_month", "months_to_legal_maturity", "zero_balance_code", "interest_rate", "pd_12m_cal"]]
      .merge(lgd_df[["loan_identifier", "observation_month", "stage", "lgd", "ead"] + d], 
             on=["loan_identifier", "observation_month"], how="inner"))

# df.to_csv("data/df.csv", index=False)

# df = pd.read_csv("data/df.csv")

# loan F15Q10288980 is 11 months overdue (months_to_legal_maturity = -11)
df["months_to_legal_maturity"] = df["months_to_legal_maturity"].clip(lower=0) 

df = df[df["zero_balance_code"] != 1]

df["hazard_factor"] = 1 - (1 - df["pd_12m_cal"])**(1/12)

df["ecl"] = np.nan
df.loc[df["stage"] == 1, "ecl"] = df["pd_12m_cal"] * df["lgd"] * df["ead"]
df.loc[df["stage"] == 2, "ecl"] = df.apply(ecl, axis=1)

# An exception that we treat separately (use ECL_12m)
df.loc[df["months_to_legal_maturity"] == 0, "ecl"] = df["pd_12m_cal"] * df["lgd"] * df["ead"]

defaults_df = pd.read_sql(
    """
    WITH history AS (
        SELECT
        loan_identifier, 
        monthly_reporting_period AS observation_month,
        current_actual_upb,
        zero_balance_code,
        LAG(current_actual_upb) OVER (
            PARTITION BY loan_identifier
            ORDER BY monthly_reporting_period)
        AS last_upb
    FROM raw.performance
    WHERE monthly_reporting_period BETWEEN 202602 AND 202603
    )

    SELECT
        loan_identifier,
        observation_month,
        last_upb
    FROM history
    WHERE zero_balance_code IN (2, 3, 9)
        AND current_actual_upb = 0
        AND last_upb IS NOT NULL
    """,
    engine
)

upb_map = defaults_df.set_index("loan_identifier")["last_upb"]

df.loc[df["stage"] == 3, "ead"] = df["loan_identifier"].map(upb_map)
df.loc[df["stage"] == 3, "ecl"] = df["lgd"] * df["ead"]

df.to_csv("data/results.csv", index=False)

df = pd.read_csv("data/results.csv")
