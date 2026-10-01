import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

"""
CHOOSE YOUR MODEL
"""

# Valid model names: mean (Historical Average LGD), xgb (XGBoost), rf (Random Forest), lgbm (LightGBM)
model_name = "xgb" 

"""
DATA SPLIT
"""

lgd_df = pd.read_parquet("data/lgd_full.csv")

lgd_df["year"] = lgd_df["default_month"] // 100

train_df = lgd_df[lgd_df["year"] <= 2023].copy()
validation_df = lgd_df[lgd_df["year"].between(2022, 2023)].copy()
test_df = lgd_df[lgd_df["year"].between(2024, 2025)].copy()

for df in [train_df, validation_df, test_df]:
    df.drop(columns=["year"], inplace=True)

target = ["lgd_model"]

numeric_features = [
    "eltv",
    "ltv",
    "cltv",
    "current_actual_upb",
    "loan_age",
    "credit_score",
    "dti",
    "interest_rate",
    "current_interest_rate"
]

categorical_features = [
    "current_loan_delinquency_status",
    "property_state"
]   

features = numeric_features + categorical_features

"""
DATA TRANSFORMATION & TRAINING
"""

X_train = train_df[features]
Y_train = train_df[target]

X_val = validation_df[features]
Y_val = validation_df[target]

X_test = test_df[features]
Y_test = test_df[target]

"""
MODELS
"""

if model_name == "mean":

    mean_lgd = np.mean(Y_train)

    Y_pred_val = mean_lgd.copy()

    Y_pred_test = mean_lgd.copy()

else:

    if model_name in ["xgb", "rf", "lgbm"]:

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

        categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                                ("onehot", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features),
                                            ("categorical", categorical_transformer, categorical_features)])

        X_train_processed = preprocessor.fit_transform(X_train)
        joblib.dump(preprocessor, "lgd/preprocessor.joblib")

        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        if model_name == "xgb":

            model = XGBRegressor(
                n_estimators=500,
                max_depth=None,
                learning_rate=0.01,
                subsample=1,
                colsample_bytree=0.9,
                tree_method="hist",
                objective="reg:squarederror",
                random_state=42
            )

        elif model_name == "rf":

            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            )

        elif model_name == "lgbm":

            model = LGBMRegressor(
                n_estimators=300,
                max_depth=-1,
                learning_rate=0.01,
                subsample=1,
                colsample_bytree=0.9,
                objective="regression",
                random_state=42
            ) 

        model.fit(X_train_processed, Y_train)
        joblib.dump(model, f"lgd/{model_name}/{model_name}.joblib")

        Y_pred_val = model.predict(X_val_processed)
        Y_pred_test = model.predict(X_test_processed)

    else:

        raise ValueError("Please provide a valid model name: mean, xgb, rf or lgbm")

validation_df["lgd_pred"] = Y_pred_val
validation_df.to_csv(f"lgd/{model_name}/{model_name}_validation_df.csv", index=False)

test_df["lgd_pred"] = Y_pred_test
test_df.to_csv(f"lgd/{model_name}/{model_name}_test_df.csv", index=False)

from functions import rmse, mae

print("VALIDATION")

print("RMSE: ", rmse(Y_val.values, Y_pred_val))
print("MAE: ", mae(Y_val.values, Y_pred_val))

print("\n")

print("TEST")

print("RMSE: ", rmse(Y_test.values, Y_pred_test))
print("MAE: ", mae(Y_test.values, Y_pred_test))