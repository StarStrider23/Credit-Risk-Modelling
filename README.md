# Credit Risk Modelling

By Alexsey Chernichenko. September 2026.

# Project Goal

## Project Goal

The goal of this project is to build an end-to-end credit risk modelling framework using Freddie Mac’s Single Family Loan Level Dataset. Data cleaning and preparation were performed in PostgreSQL, while Python was used to develop and evaluate machine learning (ML) models for 12-month PD and LGD prediction, using XGBoost, Random Forest and LightGBM. PD models were evaluated using ROC-AUC and PR-AUC, while probability calibration was assessed using Brier score and Log Loss. LGD models were evaluated using RMSE and MAE.

The resulting PD, LGD, and EAD estimates were combined to calculate ECL using a simplified staging framework. The ECL methodology is **IFRS 9-inspired** rather than a full IFRS 9 implementation, as the public dataset does not contain sufficient information to model all required contractual, behavioural and forward looking assumptions.

# Data

The project uses the Single Family Loan Level Dataset from Freddie Mac. The dataset can be accessed through the link below.

1. **Single Family Loan Level Dataset** : https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

The dataset itself continues to be free for non-commercial and academic/research as of September 2026. However, access to the dataset requires registration with Freddie Mac and acceptance of its terms of use. The dataset is not redistributed as part of this project, users interested in reproducing the analysis should obtain the data directly from Freddie Mac and comply with the applicable terms and conditions.

# Methodology 

## Methodology

The project follows an end-to-end credit risk modelling workflow covering data preparation, probability of default (PD), loss given default (LGD) and expected credit loss (ECL).

## Data preparation

The analysis is based on Freddie Mac's Single Family Loan Level Dataset, which contains loan characteristics and monthly performance information. The raw data was imported and processed using PostgreSQL. Data cleaning included handling of missing values, transforming variables into modelling ready formats, creating derived variables and constructing the target variables required for PD and LGD modelling.

A 12-month default indicator was created for the PD model. For each loan observation, the target indicates whether a default event occurred within the following 12 months. The LGD target was constructed for loans with an observed default and was based on realised losses relative to the exposure associated with the default event.

The dataset covers observations from 2015 through March of 2026. However, because the PD target represents whether a default occurs within the following 12 months, the observations after March 2025 cannot have a fully observed 12-month outcome within the available dataset. Therefore, PD (and also LGD) modelling and evaluation were restricted to observations from 2015 through March 2025.

## PD Modelling

The first modelling component estimates the probability that a loan will default within the following 12 months. Several ML models were investigated such as Logistic Regression, Random Forest, LightGBM and XGBoost. The Logistic Regression was considered to be a baseline model. 

Because default is a relatively rare event in the dataset, model discrimination was evaluated using both ROC-AUC and PR-AUC. But PR-AUC was given particular attention because it provides a more informative assessment of performance when the positive class (default class) is highly imbalanced.

The data was split temporally rather than randomly in order to better reflect a real world model development process. Earlier observations were used for model development, while later periods were reserved for validation and testing. This also allowed the model's ability to generalise to newer loan populations to be assessed. Moreover, it was decided to work with a smaller sample rather than with the full dataset simply because training ML models on the full dataset was extremely time consuming.

## PD Calibration & Model Stability

While ML models can provide strong risk ranking their raw probability estimates are not necessarily well calibrated. Logistic Regression was therefore used for calibration purposes to transform the model's raw PD estimates into probabilities that better correspond to observed default frequencies.

Calibration was evaluated using Brier score, Log Loss, calibration curves and comparisons between average predicted PD and observed default rates.

Temporal stability was also investigated by analysing changes in the distribution of model features over time. Population Stability Index (PSI) was used to quantify feature drift between the development and later populations. Features exhibiting substantial temporal instability were investigated further and alternative model specifications were compared. The final specification therefore considered both predictive performance and temporal generalisation.

## LGD modelling

The second part estimates loss given default. Realised LGD was calculated using the observed loss associated with a default event and the delinquent accrued interest. Both values are available only for defaulted loans.

The same general ML approach was considered for including XGBoost, Random Forest and LightGBM. Logistic Regression was of course not used as its purpose is to predict the probability of a categorical dependant outcome. Instead, the baseline model was a simple Historical Average Model which of course isn't a ML model per se. Because the realised LGD observations are considerably fewer and more variable than the observations available for PD modelling, data splitting was performed somewhat differently. 

## ECL calculation

The final stage combines PD, LGD and exposure at default (EAD) to estimate expected credit loss. EAD was modelled as loan's current unpaid principal balance (UPB) which is of course a simple assumption. 

A simplified staging framework was implemented to distinguish performing exposures from exposures showing credit deterioration and established default events. The staging was mostly based on each loan's delinquency status. For Stage 1 exposures, a 12-month ECL approach was used, combining the calibrated 12-month PD with LGD and EAD. For Stage 2 exposures, a simplified lifetime ECL calculation was constructed by projecting monthly default probabilities over the remaining maturities and applying discounted future losses. Stage 3 exposures were treated as defaulted exposures and assigned a simplified loss estimate based on LGD and EAD.

The final framework was applied to the March 2026 portfolio to compute ECL estimates.

This ECL calculation is intended as **IFRS 9 inspired** exercise rather a real IFRS 9 implementation. This is because the public dataset does not provide all required information. The methodology therefore focuses on demonstrating the core interaction between PD, LGD, EAD, staging and ECL using publicly available real world credit data.

# Background 

# Structure

# Results

# Discussion
