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

Finally, credit portfolios change over time. Changes in borrower characteristics, loan composition, economic conditions and portfolio ageing can cause the population on which a model is applied to differ from the population used during development. Monitoring this population/feature drift and evaluating model performance on later observations are therefore important parts of credit risk model development. In this project, temporal validation, out-of-time testing and feature stability analysis are used to investigate whether the models remain useful as the portfolio evolves.

# Structure

# Results

# Discussion
