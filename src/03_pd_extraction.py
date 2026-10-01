import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg:///" "credit_risk")

cols = ["loan_identifier", "observation_month"]

features = [
    "credit_score",
    "first_time_homebuyer_indicator",
    "mortgage_insurance_pct",
    "number_of_units",
    "occupancy_status",
    "cltv",
    "dti",
    "original_upb",
    "ltv",
    "interest_rate",
    "channel",
    "property_state",
    "property_type",
    "loan_purpose",
    "loan_term",
    "number_of_borrowers",
    "current_actual_upb",
    "loan_age",
    "months_to_legal_maturity",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "eltv",
    "delinquency_status_numeric",
    "modification_flag",
    "payment_deferral_flag"
]

target = "default_12m"

columns = ", ".join(cols + features + [target])

query = f"""
SELECT {columns}
FROM pd.model_data
"""

chunks = []

for chunk in pd.read_sql(query, engine, chunksize=100_000):
    chunks.append(chunk)

pd_df = pd.concat(chunks, ignore_index=True)

pd_df.to_parquet("data/pd.parquet", index=False)