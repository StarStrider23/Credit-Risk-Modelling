import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

"""
CHOOSE YOUR MODEL
"""

# Valid model names: lr (Linear Regression), xgb (XGBoost), rf (Random Forest), lgbm (LightGBM)
model_name = "xgb" 

"""
DATA SPLIT
"""

df = pd.read_parquet("data/pd.parquet")

df["year"] = df["observation_month"] // 100

train_df = df[df["year"] <= 2023].copy()
validation_df = df[df["year"] == 2024].copy()
test_df = df[df["year"] == 2025].copy()

for df in [train_df, validation_df, test_df]:
    df.drop(columns=["year"], inplace=True)

target = "default_12m"

train_sample, _ = train_test_split(
    train_df,
    train_size=4_000_000,
    stratify=train_df[target],
    random_state=42
)

train_sample.to_csv("pd/train_sample.csv", index=False)

validation_sample, _ = train_test_split(
    validation_df,
    train_size=800_000,
    stratify=validation_df[target],
    random_state=42
)

"""
DATA TRANSFORMATION & TRAINING
"""

X_train = train_sample.drop(columns=[target])
Y_train = train_sample[target]

X_val = validation_sample.drop(columns=[target])
Y_val = validation_sample[target]

X_test = test_df.drop(columns=[target])
Y_test = test_df[target]

numeric_features = [
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
    "current_actual_upb",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "eltv",
    "delinquency_status_numeric"
]

categorical_features = [
    "first_time_homebuyer_indicator",
    "occupancy_status",
    "channel",
    "property_state",
    "property_type",
    "loan_purpose",
    "modification_flag",
    "payment_deferral_flag"
]

if model_name == "lr":

    # For LogisticRegression
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), 
                                          ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features),
                                    ("categorical", categorical_transformer, categorical_features)])

elif model_name in ["xgb", "rf", "lgbm"]:

    # For XGBoost, RandomForest and LGBM
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), 
                                          ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features),
                                    ("categorical", categorical_transformer, categorical_features)])

else:
    raise ValueError("Please provide a valid model name: lr, xgb, rf or lgbm")
                                    

X_train_processed = preprocessor.fit_transform(X_train)
X_train_processed = X_train_processed.astype("float32")

if model_name == "lr":
    joblib.dump(preprocessor, "pd/preprocessor_lr.joblib")
elif model_name in ["xgb", "rf", "lgbm"]:
    joblib.dump(preprocessor, "pd/preprocessor.joblib")
else:
    raise ValueError("Please provide a valid model name: lr, xgb, rf or lgbm")

X_val_processed = preprocessor.transform(X_val)
X_val_processed = X_val_processed.astype("float32")

X_test_processed = preprocessor.transform(X_test)
X_test_processed = X_test_processed.astype("float32")

"""
MODELS
"""

if model_name == "lr":

    model = LogisticRegression(solver="liblinear",
        max_iter=500, 
        class_weight="balanced",
        random_state=42)

elif model_name == "xgb":

    model = XGBClassifier(n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42)

elif model_name == "rf":    

    model = RandomForestClassifier(n_estimators=500, 
        max_depth=None, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        max_features="log2",
        class_weight="balanced",
        bootstrap=True,
        random_state=42, 
        n_jobs=-1)

elif model_name == "lgbm":

    model = LGBMClassifier(boosting_type="gbdt",
        n_estimators=500,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        colsample_bytree=0.75,
        learning_rate=0.1,
        reg_alpha=0,
        reg_lambda=0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        force_row_wise=True)

else:
    raise ValueError("Please provide a valid model name: lr, xgb, rf or lgbm")

"""
FITTING & PREDICTION
"""

model.fit(X_train_processed, Y_train)
joblib.dump(model, f"pd/{model_name}/{model_name}.joblib")

Y_val_prob = model.predict_proba(X_val_processed)[:, 1]
validation_sample["pd_12m"] = Y_val_prob
validation_sample.to_csv(f"pd/{model_name}/{model_name}_validation_df.csv", index=False)

Y_test_prob = model.predict_proba(X_test_processed)[:, 1]
test_df["pd_12m"] = Y_test_prob
test_df.to_csv(f"pd/{model_name}/{model_name}_test_df.csv", index=False)

from sklearn.metrics import roc_auc_score, average_precision_score

print("VALIDATION")
print("ROC AUC: ", roc_auc_score(Y_val, Y_val_prob))
print("PR AUC: ", average_precision_score(Y_val, Y_val_prob))
print("\n")
print("TEST")
print("ROC AUC: ", roc_auc_score(Y_test, Y_test_prob))
print("PR AUC: ", average_precision_score(Y_test, Y_test_prob))