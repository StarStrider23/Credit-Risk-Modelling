# REPORT STATUS: ALMOST FINISHED

# Credit Risk Modelling

By Alexsey Chernichenko. September 2026.

# Project Goal

## Project Goal

The goal of this project is to build a credit risk modelling pipeline using Freddie Mac’s Single Family Loan Level Dataset. Data cleaning and preparation were performed in PostgreSQL while Python was used to develop and evaluate machine learning (ML) models for 12-month Probability Default (PD) and Loss Given Default (LGD) prediction, using XGBoost, Random Forest and LightGBM. PD models were evaluated using ROC-AUC and PR-AUC, while probability calibration was assessed using Brier score and Log Loss. LGD models were evaluated using RMSE and MAE.

The resulting PD and LGD estimates were then combined to calculate ECL using a simplified staging framework and Exposure At Default (EAD) assumption. The ECL methodology is **IFRS 9 inspired** rather than a full IFRS 9 implementation since the public dataset does not contain sufficient information to model all required information.

# Data

The project uses the Single Family Loan Level Dataset from Freddie Mac. The dataset can be accessed through the link below.

1. **Single Family Loan Level Dataset** : https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

The dataset itself continues to be free for non-commercial and academic/research as of September 2026. However, access to the dataset requires registration with Freddie Mac and acceptance of its terms of use. The dataset is not redistributed as part of this project, users interested in reproducing the analysis should obtain the data directly from Freddie Mac and comply with the applicable terms and conditions.

# Methodology 

The project follows a credit risk modelling workflow covering data preparation, PD, LGD and ECL.

## Data preparation

The Freddie Mac's Single Family Loan Level Dataset contains loan characteristics and monthly performance information. The raw data was imported and processed using PostgreSQL. Data cleaning included handling of missing values, transforming variables into modelling ready formats, creating derived variables and constructing the target variables required for PD and LGD modelling.

A 12-month default indicator was created for the PD model. For each loan observation, the target indicates whether a default event occurred within the following 12 months. The LGD target was constructed for loans with an observed default and was based on realised losses relative to the exposure associated with the default event.

The dataset covers observations from 2015 through March of 2026. However, because the PD target represents whether a default occurs within the following 12 months, the observations after March 2025 cannot have a fully observed 12-month outcome. Therefore, PD (and also LGD) modelling and evaluation were restricted to observations from 2015 through March 2025.

## PD Modelling

The first modelling component estimates the probability that a loan will default within the following 12 months. Several ML models were investigated such as Logistic Regression, Random Forest, LightGBM and XGBoost. The Logistic Regression was considered to be a baseline model. 

Because default is a rare event in the dataset, model performance was assessed through both ROC-AUC and PR-AUC. But PR-AUC was given particular attention because it provides a more informative assessment of performance when the positive class (default class) is highly imbalanced.

The data was split temporally rather than randomly in order to better reflect a real world model development process. Earlier observations were used for model development, while later periods were reserved for validation and testing. This also allowed the model's ability to generalise to newer loan populations to be assessed. Moreover, it was decided to work with a smaller sample rather than with the full dataset simply because training ML models on the full dataset turned out to be extremely time consuming.

## PD Calibration & Model Stability

While ML models can provide strong risk ranking their raw probability estimates are not necessarily well calibrated. Logistic Regression was therefore used for calibration purposes to transform the model's raw PD estimates into probabilities that better correspond to observed default frequencies.

Calibration was evaluated using Brier score, Log Loss, calibration curves and comparisons between average predicted PD and observed default rates.

Temporal stability was also investigated by analysing changes in the distribution of model features over time. Population Stability Index (PSI) was used to quantify feature drift between the development and later populations. Features exhibiting substantial temporal instability were investigated further and alternative model specifications were compared. The final specification therefore considered both predictive performance and temporal generalisation.

## LGD modelling

The second part estimates loss given default. Realised LGD was calculated using the observed loss associated with a default event, the remaining unpaid principal balance (UPB) and the delinquent accrued interest. All provided by Freddie Mac in the dataset and available only for defaulted loans.

The same general ML approach was considered for including XGBoost, Random Forest and LightGBM. Logistic Regression was of course not used as its purpose is to predict the probability of a categorical dependant outcome. Instead, the baseline model was a simple Historical Average Model which of course isn't a ML model per se. Because the realised LGD observations are considerably fewer and more variable than the observations available for PD modelling, data splitting was performed somewhat differently. 

## ECL calculation

The final stage combines PD, LGD and EAD to estimate expected credit loss. Unlike EAD for the realised LGD, which accounted for loan's current UPB and delinquency accrued interest, EAD estimate was modelled simply current UPB. This is of course due to delinquency accrued interest being available for defaults only.

A simplified staging framework was implemented to distinguish performing exposures from exposures showing credit deterioration and established default events. The staging was mostly based on each loan's delinquency status. For Stage 1 exposures, a 12-month ECL approach was used, combining the calibrated 12-month PD with LGD and EAD. For Stage 2 exposures, a simplified lifetime ECL calculation was constructed by projecting monthly default probabilities over the remaining maturities and applying discounted future losses. Stage 3 exposures were treated as defaulted exposures and assigned a simplified loss estimate based on LGD and EAD.

The final framework was applied to the March 2026 portfolio to compute ECL estimates.

This ECL calculation is intended as **IFRS 9 inspired** exercise rather a real IFRS 9 implementation. This is because the public dataset does not provide all required contractual, behavioural and forward looking assumptions. The methodology therefore focuses on demonstrating the core interaction between PD, LGD, EAD, staging and ECL using publicly available real world credit data.

# Background 

Credit risk modelling is concerned with estimating the potential loss that a financial institution may incur when a borrower fails to meet their contractual obligations. A common framework for quantifying expected credit losses is based on three key components: PD, LGD and EAD.

## PD 

PD represents the likelihood that a borrower will default within a given time horizon. In this project, the primary target is the probability of default within the following 12 months:

$$
PD_{12m} = P(\text{default within 12 months})
$$

## LGD

LGD represents the proportion of the exposure that is expected to be lost if a default occurs. It reflects the amount that cannot be recovered through collateral, repayments or other recovery processes. The Freddie Mac's dataset unfortunately does not provide LGD. However, it does provide necessary ingredients for construction of realised LGD.

$$
LGD = \frac{\text{Actual Loss}}{\text{Zero Balance Removal UPB} + \text{Delinquent Accrued Interest}}
$$

## EAD

EAD represents the amount of exposure outstanding when a default occurs. For a mortgage portfolio, this is primarily related to the outstanding loan balance. This project constructs realised EAD as:

$$
EAD = \text{Zero Balance Removal UPB} + \text{Delinquent Accrued Interest}
$$

Whereas the EAD estimate is modelled simply as:

$$
EAD = \text{Zero Balance Removal UPB}
$$

## DF

Discount Factor (DF) is the final ingredient required for ECL calculation. It adjusts future cash shortfalls to their present value at the reporting date to reflect the time value of money. The discount factor requires loan's Effective Interest Rate at origination. The dataset provides original interest rate for every loan. However, it is unknown whether the rate is credit adjusted. Anyhow, the discounted factor is calculated as

$$
DF_t = \frac{1}{(1 + r)^t}
$$

Where $r$ is interest rate at origination and $t$ is a given future time horizon.

## ECL

These components can be then combined to estimate expected credit loss. In its simplest form, the expected loss for an exposure can be expressed as:

$$
ECL_t = \sum_t PD_t \times LGD_t \times EAD_t \times DF_t
$$

The time horizon is important when calculating ECL. A 12-month ECL estimate considers defaults occurring over the following 12 months whereas a lifetime ECL estimate considers potential losses over the remaining expected lifetime of the exposure. 

## Hazard factor

As the ECL calculation requires PD estimates across all time horizons, an additional tool is required. To estimate PD across all time horizons, the 12-month PD is converted into a monthly hazard factor. This hazard factor represents the conditional probability of default in a given month, assuming the loan has survived up to that point. Assuming a constant monthly hazard, the factor can be derived from the 12-month PD as:

$$
h = 1-(1-PD_{12m})^{1/12}
$$

The probability of default in month $t$ is then:

$$
p_t = h(1-h)^{t-1}
$$

This probability should be interpreted as loan's conditional probability of not defaulting $(1-h)$ during the first $(t-1)$ months followed by a default with conditional probability $h$ during the last month. The corresponding cumulative probability of default by month \(t\) is:

$$
PD_t = 1-(1-h)^t
$$

These monthly probabilities allow the 12-month PD estimate to be extended over the remaining contractual maturity of an exposure when calculating a simplified lifetime ECL.

## Staging Framework

Credit risk is also commonly divided into different stages based on changes in credit risk. Under the IFRS 9 framework, Stage 1 generally represents exposures that have not experienced a significant increase in credit risk and are associated with 12-month expected credit losses. Stage 2 represents exposures where credit risk has increased significantly and lifetime expected credit losses are recognised. Stage 3 represents credit-impaired or defaulted exposures.

In this project, the stages are constructed as following: Stage 1 covers the loans that have missed a payment or fallen behind on a financial obligation by 30 days or less, Stage 2 includes the loans whose delinquency status is longer than 30 days and the Stage 3 loans are the defaulted loans. 

## Credit Risk Model Evaluation

A key challenge in credit risk modelling is that defaults are relatively rare compared with non-defaults. This creates a highly imbalanced classification problem and makes model evaluation based solely on accuracy inappropriate. Metrics such as ROC-AUC and particularly Precision-Recall AUC (PR-AUC) are therefore useful for assessing a model's ability to distinguish between higher and lower risk exposures.

Another important distinction is between discrimination and calibration. A model can rank borrowers effectively, meaning that higher predicted PDs generally correspond to higher default risk, but still produce probabilities that are too high or too low. This distinction is important when model outputs are used directly in ECL calculations since the absolute level of PD affects the estimated loss.

Finally, credit portfolios change over time. Changes in borrower characteristics, loan composition and its ageing can cause the population on which a model is applied to differ from the population used during development. Monitoring this population/feature drift and evaluating model performance on later observations are therefore important parts of credit risk model development. 

# Structure

# Results

## PD Modelling

For the PD Modelling, the dataset was splitted in the following way: training - from 2015 to 2023, validation - 2024 and test - January through March 2025. 

For each of the four models, the results are divided into three subsections - Validation and Test Set results as well as Monthly Performance. In each of the first two subsections, there are two plots - ROC-AUC and PR-AUC curve plots. There is also a table with metrics that will be taken into account upon model evaluation. Finally, the Monthly Performance subsection demonstrates how the evaluation metrics evolve across the validation and test set month-by-month.

### Logistic Regression

#### Validation Set

<img width="1258" height="763" alt="Снимок экрана — 2026-09-30 в 16 10 50" src="https://github.com/user-attachments/assets/473b89c0-4aa8-4ee9-bee2-669c84974055" />

<img width="1259" height="740" alt="Снимок экрана — 2026-09-30 в 15 41 12" src="https://github.com/user-attachments/assets/9571c283-4a79-4923-9cee-a175a94106a2" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.943 |
| PR-AUC  | 0.0765 |
| Brier Score | 0.000293 |
| Log Loss | 0.130 |
| Average Predicted Default Rate | 0.0746 |
| Actual Default Rate | 0.000398 |

#### Test Set

<img width="1262" height="767" alt="Снимок экрана — 2026-09-30 в 14 59 09" src="https://github.com/user-attachments/assets/9e9c28f0-e14a-4322-995e-6ebf90232083" />

<img width="1259" height="765" alt="Снимок экрана — 2026-09-30 в 15 42 43" src="https://github.com/user-attachments/assets/e67433c2-2a57-468f-ba49-1c71df55e8e7" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.933 |
| PR-AUC  | 0.0504 |
| Brier Score | 0.000284 |
| Log Loss | 0.126 |
| Average Predicted Default Rate | 0.0722 |
| Actual Default Rate | 0.000312 |

#### Monthly Predictions

| Month   | Defaults | Observations | Avg. Predicted PD | Actual Default Rate | ROC-AUC | PR-AUC |
| ------- | -------- | ------------ | ----------------- | ------------------- | ------- | ------ |
| 2024-01 |       29 |       68761  |             7.94% |             0.0422% |   0.975 |  0.072 |
| 2024-02 |       23 |       68268  |             7.87% |             0.0337% |   0.975 |  0.104 |
| 2024-03 |       32 |       68173  |             7.86% |             0.0469% |   0.956 |  0.083 |
| 2024-04 |       31 |       67824  |             7.68% |             0.0457% |   0.962 |  0.103 |
| 2024-05 |       32 |       67391  |             7.42% |             0.0475% |   0.949 |  0.084 |
| 2024-06 |       30 |       67035  |             7.31% |             0.0448% |   0.930 |  0.087 |
| 2024-07 |       23 |       66369  |             7.27% |             0.0347% |   0.938 |  0.058 |
| 2024-08 |       21 |       66129  |             7.24% |             0.0318% |   0.907 |  0.041 |
| 2024-09 |       22 |       65540  |             7.33% |             0.0336% |   0.951 |  0.094 |
| 2024-10 |       27 |       65444  |             7.24% |             0.0413% |   0.923 |  0.096 |
| 2024-11 |       24 |       65042  |             7.19% |             0.0369% |   0.932 |  0.068 |
| 2024-12 |       21 |       64024  |             7.16% |             0.0328% |   0.894 |  0.098 |
| 2025-01 |       33 |       92410  |             7.29% |             0.0357% |   0.933 |  0.081 |
| 2025-02 |       30 |       91773  |             7.26% |             0.0327% |   0.926 |  0.048 |
| 2025-03 |       23 |       91075  |             7.10% |             0.0253% |   0.941 |  0.030 |

### XGBoost

#### Validation Set

<img width="1235" height="749" alt="Снимок экрана — 2026-09-30 в 14 56 57" src="https://github.com/user-attachments/assets/7b5a279a-d772-484c-974c-1934cc0b31df" />

<img width="1250" height="733" alt="Снимок экрана — 2026-09-30 в 14 57 52" src="https://github.com/user-attachments/assets/1279c9bb-219a-42df-8313-048268e3be83" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.948 |
| PR-AUC  | 0.351 |
| Brier Score | 0.000326 |
| Log Loss | 0.00243 |
| Average Predicted Default Rate | 8.41e-5 |
| Actual Default Rate | 3.98e-4 |

#### Test Set

<img width="1262" height="767" alt="Снимок экрана — 2026-09-30 в 14 59 09" src="https://github.com/user-attachments/assets/0a07ba0c-170b-44c6-a0c3-aa1efa540106" />

<img width="1241" height="755" alt="Снимок экрана — 2026-09-30 в 14 58 30" src="https://github.com/user-attachments/assets/808ed2cd-3a20-4b13-82f5-2755bfe5b521" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.938 |
| PR-AUC  | 0.189 |
| Brier Score | 0.000288 |
| Log Loss | 0.00229 |
| Average Predicted Default Rate | 5.13e-5 |
| Actual Default Rate | 3.12e-4 |

#### Monthly Predictions

| Month   | Defaults | Observations | Avg. Predicted PD | Actual Default Rate | ROC-AUC | PR-AUC |
| ------- | -------- | ------------ | ----------------- | ------------------- | ------- | ------ |
| 2024-01 |       29 |       68761  |           0.0162% |             0.0422% |   0.982 |  0.613 |
| 2024-02 |       23 |       68268  |           0.0108% |             0.0337% |   0.983 |  0.533 |
| 2024-03 |       32 |       68173  |           0.0101% |             0.0469% |   0.960 |  0.373 |
| 2024-04 |       31 |       67824  |           0.0087% |             0.0457% |   0.962 |  0.433 |
| 2024-05 |       32 |       67391  |           0.0084% |             0.0475% |   0.946 |  0.314 |
| 2024-06 |       30 |       67035  |           0.0083% |             0.0448% |   0.935 |  0.301 |
| 2024-07 |       23 |       66369  |           0.0071% |             0.0347% |   0.939 |  0.346 |
| 2024-08 |       21 |       66129  |           0.0062% |             0.0318% |   0.941 |  0.228 |
| 2024-09 |       22 |       65540  |           0.0075% |             0.0336% |   0.938 |  0.350 |
| 2024-10 |       27 |       65444  |           0.0064% |             0.0413% |   0.921 |  0.297 |
| 2024-11 |       24 |       65042  |           0.0049% |             0.0369% |   0.938 |  0.158 |
| 2024-12 |       21 |       64024  |           0.0056% |             0.0328% |   0.918 |  0.255 |
| 2025-01 |       33 |       92410  |           0.0058% |             0.0357% |   0.937 |  0.234 |
| 2025-02 |       30 |       91773  |           0.0052% |             0.0327% |   0.934 |  0.172 |
| 2025-03 |       23 |       91075  |           0.0044% |             0.0253% |   0.944 |  0.163 |

### Random Forest

#### Validation Set

<img width="1264" height="784" alt="Снимок экрана — 2026-09-30 в 15 52 51" src="https://github.com/user-attachments/assets/7aab39ab-90be-47ca-96c7-1c084f971386" />

<img width="1258" height="750" alt="Снимок экрана — 2026-09-30 в 15 53 14" src="https://github.com/user-attachments/assets/5472e3d6-b52d-415d-881d-03f592b7bc0c" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.933 |
| PR-AUC  | 0.383 |
| Brier Score | 0.000355 |
| Log Loss | 0.00436 |
| Average Predicted Default Rate | 3.08e-3 |
| Actual Default Rate | 3.94e-4 |

#### Test Set

<img width="1251" height="740" alt="Снимок экрана — 2026-09-30 в 15 53 41" src="https://github.com/user-attachments/assets/662f312e-43af-49e5-9cbc-41490d2521f3" />

<img width="1249" height="739" alt="Снимок экрана — 2026-09-30 в 15 54 05" src="https://github.com/user-attachments/assets/65e72f2c-09dc-45f3-896c-c19287ad19b6" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.881 |
| PR-AUC  | 0.0748 |
| Brier Score | 0.000380 |
| Log Loss | 0.00509 |
| Average Predicted Default Rate | 3.08e-3 |
| Actual Default Rate | 3.12e-4 |

#### Monthly Predictions

| Month   | Defaults | Observations | Avg. Predicted PD | Actual Default Rate | ROC-AUC | PR-AUC |
| ------- | -------- | ------------ | ----------------- | ------------------- | ------- | ------ |
| 2024-01 |       29 |       68761  |           0.3255% |             0.0422% |   0.980 |  0.700 |
| 2024-02 |       23 |       68268  |           0.3191% |             0.0337% |   0.989 |  0.557 |
| 2024-03 |       32 |       68173  |           0.3178% |             0.0469% |   0.961 |  0.495 |
| 2024-04 |       31 |       67824  |           0.3139% |             0.0457% |   0.967 |  0.481 |
| 2024-05 |       32 |       67391  |           0.2999% |             0.0475% |   0.931 |  0.426 |
| 2024-06 |       30 |       67035  |           0.3072% |             0.0448% |   0.924 |  0.389 |
| 2024-07 |       23 |       66369  |           0.3020% |             0.0347% |   0.891 |  0.343 |
| 2024-08 |       21 |       66129  |           0.2991% |             0.0318% |   0.929 |  0.235 |
| 2024-09 |       22 |       65540  |           0.3000% |             0.0336% |   0.890 |  0.266 |
| 2024-10 |       27 |       65444  |           0.3044% |             0.0413% |   0.889 |  0.261 |
| 2024-11 |       24 |       65042  |           0.3048% |             0.0369% |   0.914 |  0.060 |
| 2024-12 |       21 |       64024  |           0.3044% |             0.0328% |   0.902 |  0.107 |
| 2025-01 |       33 |       92410  |           0.3095% |             0.0357% |   0.893 |  0.116 |
| 2025-02 |       30 |       91773  |           0.3115% |             0.0327% |   0.878 |  0.066 |
| 2025-03 |       23 |       91075  |           0.3019% |             0.0253% |   0.867 |  0.077 |

### LightGBM

#### Validation Set

<img width="1249" height="772" alt="Снимок экрана — 2026-09-30 в 16 02 04" src="https://github.com/user-attachments/assets/6a7252e7-db83-45e2-b6ff-6f349cc41a78" />

<img width="1254" height="756" alt="Снимок экрана — 2026-09-30 в 16 02 24" src="https://github.com/user-attachments/assets/b2c2c266-1a83-473f-8d2e-66c4daf62c82" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.955 |
| PR-AUC  | 0.380 |
| Brier Score | 0.000654 |
| Log Loss | 0.0033 |
| Average Predicted Default Rate | 1.65e-3 |
| Actual Default Rate | 3.94e-4 |

#### Test Set

<img width="1256" height="751" alt="Снимок экрана — 2026-09-30 в 16 02 43" src="https://github.com/user-attachments/assets/b938109c-a50b-4e7e-9eed-a964806c67df" />

<img width="1262" height="776" alt="Снимок экрана — 2026-09-30 в 16 03 18" src="https://github.com/user-attachments/assets/8b7931a5-1c26-43bf-b879-d70215278ccd" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.931 |
| PR-AUC  | 0.172 |
| Brier Score | 0.000724 |
| Log Loss | 0.0036 |
| Average Predicted Default Rate | 1.65e-3 |
| Actual Default Rate | 3.12e-4 |

#### Monthly Predictions

| Month   | Defaults | Observations | Avg. Predicted PD | Actual Default Rate | ROC-AUC | PR-AUC |
| ------- | -------- | ------------ | ----------------- | ------------------- | ------- | ------ |
| 2024-01 |       29 |       68761  |           0.1982% |             0.0422% |   0.977 |  0.614 |
| 2024-02 |       23 |       68268  |           0.1858% |             0.0337% |   0.979 |  0.451 |
| 2024-03 |       32 |       68173  |           0.1887% |             0.0469% |   0.963 |  0.442 |
| 2024-04 |       31 |       67824  |           0.1722% |             0.0457% |   0.969 |  0.466 |
| 2024-05 |       32 |       67391  |           0.1549% |             0.0475% |   0.967 |  0.389 |
| 2024-06 |       30 |       67035  |           0.1608% |             0.0448% |   0.944 |  0.373 |
| 2024-07 |       23 |       66369  |           0.1493% |             0.0347% |   0.964 |  0.417 |
| 2024-08 |       21 |       66129  |           0.1531% |             0.0318% |   0.928 |  0.227 |
| 2024-09 |       22 |       65540  |           0.1543% |             0.0336% |   0.954 |  0.406 |
| 2024-10 |       27 |       65444  |           0.1469% |             0.0413% |   0.919 |  0.371 |
| 2024-11 |       24 |       65042  |           0.1533% |             0.0369% |   0.956 |  0.150 |
| 2024-12 |       21 |       64024  |           0.1611% |             0.0328% |   0.926 |  0.252 |
| 2025-01 |       33 |       92410  |           0.1703% |             0.0357% |   0.934 |  0.232 |
| 2025-02 |       30 |       91773  |           0.1697% |             0.0327% |   0.927 |  0.147 |
| 2025-03 |       23 |       91075  |           0.1554% |             0.0253% |   0.933 |  0.144 |

## LGD Modelliing

For the LGD Modelling, the dataset was splitted in the following way: training - from 2015 to 2021, validation - from 2022 to 2023 and test - from 2024 to March 2025.

For each of the four models, the results are divided into three subsections - Validation and Test Set results as well as Annual Performance. In each of the first two subsections, there is a plot - Realised vs Actual LGD. There is also a table with metrics that will be taken into account upon model evaluation. Finally, the Annual Performance subsection demonstrates how the evaluation metrics evolve across the validation and test set year-by-year.

The Realised vs Actual LGD plot should be regarded as following: each scatter point represents a loan. If the point lies on the reference line, the predicted LGD is exactly right. The point being above/under the line means that the LGD estimate was higher/lower than the actual LGD.

### Historical Mean Average

#### Validation Set

<img width="1277" height="788" alt="Снимок экрана — 2026-09-30 в 17 35 56" src="https://github.com/user-attachments/assets/53366230-9b2f-4fa8-bf12-6d9a8f3f003f" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.270 |
| Mean Actual LGD  | 0.194 |
| RMSE | 0.270 |
| MAE | 0.242 |
| Spearman Correlation | NaN |
| Pearson Correlation | NaN |

#### Test Set

<img width="1251" height="747" alt="Снимок экрана — 2026-09-30 в 17 36 14" src="https://github.com/user-attachments/assets/c79b9315-62f4-43a4-8a27-c34f57ad2003" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.270 |
| Mean Actual LGD  | 0.290 |
| RMSE | 0.361 |
| MAE | 0.297 |
| Spearman Correlation | NaN |
| Pearson Correlation | NaN |

#### Annual Performance

| Year | Average LGD | Average LGD Prediction |  RMSE |   MAE |
| ---- | ----------- | ---------------------- | ----- | ----- |
| 2021 |       0.039 |                  0.270 | 0.233 | 0.231 |
| 2022 |       0.188 |                  0.270 | 0.263 | 0.234 |
| 2023 |       0.229 |                  0.270 | 0.279 | 0.248 |
| 2024 |       0.281 |                  0.270 | 0.344 | 0.281 |
| 2025 |       0.301 |                  0.270 | 0.390 | 0.327 |

### XGBoost

#### Validation Set

<img width="1237" height="778" alt="Снимок экрана — 2026-09-30 в 16 41 48" src="https://github.com/user-attachments/assets/fa497ff8-b5ef-42ab-86c9-29b4a61d8631" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.227 |
| Mean Actual LGD  | 0.194 |
| RMSE | 0.000724 |
| MAE | 0.0036 |
| Spearman Correlation | 0.608 |
| Pearson Correlation | 0.641 |

#### Test Set

<img width="1240" height="741" alt="Снимок экрана — 2026-09-30 в 16 42 08" src="https://github.com/user-attachments/assets/3954565a-ab01-4dc3-9d50-183e4a5bad31" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.252 |
| Mean Actual LGD  | 0.290 |
| RMSE | 0.331|
| MAE | 0.243 |
| Spearman Correlation | 0.394 |
| Pearson Correlation | 0.421 |

### Annual Performance

| Year |  Average LGD | Average LGD Prediction |  rmse  |  mae  |
| ---- | ------------ | ---------------------- | ------ | ----- |
| 2021 |   0.0394     |    0.114               |  0.093 | 0.074 |
| 2022 |   0.188      |    0.233               |  0.182 | 0.134 |
| 2023 |   0.229      |    0.239               |  0.230 | 0.158 |
| 2024 |   0.281      |    0.290               |  0.321 | 0.244 |
| 2025 |   0.301      |    0.199               |  0.348 | 0.248 |

### Random Forest

#### Validation Set

<img width="1252" height="775" alt="Снимок экрана — 2026-09-30 в 17 50 25" src="https://github.com/user-attachments/assets/a4d829e9-6052-4315-bb5f-68606d9c9f59" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.261 |
| Mean Actual LGD  | 0.194 |
| RMSE | 0.215 |
| MAE | 0.183 |
| Spearman Correlation | 0.625 |
| Pearson Correlation | 0.692 |

#### Test Set

<img width="1271" height="761" alt="Снимок экрана — 2026-09-30 в 17 50 45" src="https://github.com/user-attachments/assets/a907e09f-1380-419d-b499-7baae00173d9" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.270 |
| Mean Actual LGD  | 0.290 |
| RMSE | 0.324 |
| MAE | 0.264 |
| Spearman Correlation | 0.439 |
| Pearson Correlation | 0.508 |

#### Annual Performance

| Year | Average LGD | Average LGD Prediction |  RMSE |   MAE |
| ---- | ----------- | ---------------------- | ----- | ----- |
| 2021 |       0.039 |                  0.204 | 0.190 | 0.165 |
| 2022 |       0.188 |                  0.261 | 0.200 | 0.171 |
| 2023 |       0.229 |                  0.270 | 0.233 | 0.196 |
| 2024 |       0.281 |                  0.283 | 0.310 | 0.260 |
| 2025 |       0.301 |                  0.252 | 0.348 | 0.279 |

### LightGBM

#### Validation Set

<img width="1260" height="787" alt="Снимок экрана — 2026-09-30 в 17 55 44" src="https://github.com/user-attachments/assets/b3b81018-a767-45dd-b269-e1a6549cf92d" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.230 |
| Mean Actual LGD  | 0.194 |
| RMSE | 0.193 |
| MAE | 0.137 |
| Spearman Correlation | 0.652 |
| Pearson Correlation | 0.682 |

#### Test Set

<img width="1258" height="756" alt="Снимок экрана — 2026-09-30 в 17 56 01" src="https://github.com/user-attachments/assets/ad19c881-8f0e-451a-aa40-8500e91aea46" />

|         |       |
| ------- | ----- |
| Mean Predicted LGD | 0.255 |
| Mean Actual LGD  | 0.290 |
| RMSE | 0.330 |
| MAE | 0.256 |
| Spearman Correlation | 0.330 |
| Pearson Correlation | 0.414 |

#### Annual Performance

| Year | Average LGD | Average LGD Prediction |  RMSE |   MAE |
| ---- | ----------- | ---------------------- | ----- | ----- |
| 2021 |       0.039 |                  0.105 | 0.095 | 0.066 |
| 2022 |       0.188 |                  0.236 | 0.164 | 0.124 |
| 2023 |       0.229 |                  0.245 | 0.230 | 0.161 |
| 2024 |       0.281 |                  0.279 | 0.307 | 0.243 |
| 2025 |       0.301 |                  0.218 | 0.364 | 0.280 |

## XGBoost PD Calibration

Below are the results of calibrating raw XGBoost Test (January - March 2025) probabilities on the Validation Set (2024). The graph shows the calibration plot. The reference line represents a perfect calibration. The calibration curve being below the reference line imply that the calibrated probabilities are lower than the actual default rate and the model underpredicts it. 

<img width="1300" height="775" alt="Снимок экрана — 2026-09-30 в 18 52 25" src="https://github.com/user-attachments/assets/4db4a25b-8c89-450c-8048-cd84275db753" />

The first table below demonstrates how the raw vs calibrated probabilities as well as other metrics. Notice how the ROC-AUC and PR-AUC remain unchanged while the Brier Score and Log Loss decrease.

|                    | ROC-AUC | PR-AUC | Brier Score | Log Loss | Actual Default Rate | Average Predicted Default Rate |
| ------------------ | ------- | ------ | ----------- | -------- | ------------------- | ------------------------------ |
| Before Calibration | 0.938   | 0.189  | 0.000288    | 0.00229  | 0.000312            | 5.12e-5                        |
| After Calibration  | 0.938   | 0.189. | 0.000283    | 0.00189  | 0.000312            | 0.000283                       |

The second table shows how the raw and calibrated probabilities vary month-by-month.

| Month   | Defaults | Observations | Avg. Predicted PD | Avg. Predicted Calibrated PD | Actual Default Rate | 
| ------- | -------- | ------------ | ----------------- |----------------------------- | ------------------- |
| 2025-01 |       33 |       92410  |           0.0058% |    0.0292%                   |             0.0422% |
| 2025-02 |       30 |       91773  |           0.0052% |    0.0288%                   |             0.0337% |
| 2025-03 |       23 |       91075  |           0.0044% |    0.0270%                   |             0.0469% |

## Reestimated XGBoost PD Models

For the reestimated PD Modelling, the previous training set was extended up 2024. For the reestimated LGD Modelling, the previous training set also absorbed the validations set and covers 2015 - 2023. The corresponding test sets remained unchanged and the new results are presented below. For this part, only XGBoost was considered.

<img width="1245" height="764" alt="Снимок экрана — 2026-09-30 в 18 08 16" src="https://github.com/user-attachments/assets/5594fe2c-4bf9-4617-a5f1-00e5a6583418" />

<img width="1248" height="782" alt="Снимок экрана — 2026-09-30 в 18 08 55" src="https://github.com/user-attachments/assets/4a8899d3-a3dc-4473-85a3-0ba40425c19a" />

|         |       |
| ------- | ----- |
| ROC-AUC | 0.980 |
| PR-AUC  | 0.624 |
| Average Predicted Default Rate | 1.30e-4 |
| Actual Default Rate | 3.12e-4 |

| Month   | Defaults | Observations | Avg. Predicted PD | Actual Default Rate | ROC-AUC | PR-AUC |
| ------- | -------- | ------------ | ----------------- | ------------------- | ------- | ------ |
| 2025-01 |       33 |       92410  |           0.0158% |             0.0357% |   0.984 |  0.745 |
| 2025-02 |       30 |       91773  |           0.0133% |             0.0327% |   0.980 |  0.588 |
| 2025-03 |       23 |       91075  |           0.0101% |             0.0253% |   0.975 |  0.511 |

## ECL



# Discussion

## PD Modelling

- PSI results (loan_age 5.95, months_to_legal_maturity 7.78 and eltv 2.34 with much high PSI, the first 2 removed and results got better while removing eltv caused significant drop in results. The other numeric features PSI between 0 and 0.5, while even <0.25 considered to be large, removal of this metrics caused only deterioration)
