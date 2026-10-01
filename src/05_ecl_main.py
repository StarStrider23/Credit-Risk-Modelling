import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/results.csv")

# print(df["ecl"].describe())

# plt.figure(figsize=(12,8))

# plt.hist(df["ecl"], density=True)

# plt.show()

df["pd_decile"] = pd.qcut(df["pd_12m_cal"], 10, labels=False, duplicates="drop") + 1

f = df.groupby("pd_decile").agg(
    mean_pd=("pd_12m_cal", "mean"),
    mean_credit_score=("credit_score", "mean"),
    mean_dti=("dti", "mean"),
    mean_ltv=("ltv", "mean"),
    mean_eltv=("eltv", "mean"),
    mean_loan_age=("loan_age", "mean"),
    mean_ead=("ead", "mean"))

print(f)

df["ecl_share"] = df["ecl"] / df["ecl"].sum()

ecl_by_pd = df.groupby("pd_decile").agg(loans=("loan_identifier", "count"),
                                        mean_pd=("pd_12m_cal", "mean"),
                                        mean_lgd=("lgd", "mean"),
                                        mean_ead=("ead", "mean"),
                                        mean_ecl=("ecl", "mean"),
                                        total_ecl=("ecl", "sum"),
                                        ecl_share=("ecl_share", "sum"),
                                        stage_mean=("stage", "mean")
)

print(ecl_by_pd)

ecl_by_stage = df[df["pd_decile"] == 10].groupby("stage").agg(loans=("loan_identifier", "count"),
                                    total_ead=("ead", "sum"),
                                    total_ecl=("ecl", "sum"),
                                    mean_ecl=("ecl", "mean"),
                                    mean_pd=("pd_12m_cal", "mean")
)

print(ecl_by_stage)