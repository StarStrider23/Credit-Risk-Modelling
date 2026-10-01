import pandas as pd
from functions import calculate_psi

model_name = "xgb"

train_df = pd.read_csv("pd/train_sample.csv")
val_df = pd.read_csv(f"pd/{model_name}/{model_name}_validation_df.csv")
test_df = pd.read_csv(f"pd/{model_name}/{model_name}_test_df.csv")

features = [
    "credit_score",
    "mortgage_insurance_pct",
    "number_of_units",
    "cltv",
    "dti",
    "original_upb",
    "ltv",
    "interest_rate",
    "loan_term",
    "number_of_borrowers",
    "loan_age",
    "months_to_legal_maturity",
    "current_actual_upb",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "eltv",
    "delinquency_status_numeric"
]

reference = train_df

drift_results = []

for feature in features:

    psi = calculate_psi(reference[feature], test_df[feature])
    drift_results.append({"feature": feature, "psi": psi})

drift_results = pd.DataFrame(drift_results)
drift_results.sort_values("psi", ascending=False)

print(drift_results)