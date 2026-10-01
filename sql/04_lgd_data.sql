CREATE SCHEMA IF NOT EXISTS lgd;

DROP TABLE IF EXISTS lgd.lgd_data;
DROP TABLE IF EXISTS lgd.origination_data;

CREATE TABLE lgd.lgd_data AS
SELECT
    loan_identifier,
    monthly_reporting_period AS observation_month,
    zero_balance_code,
    zero_balance_effective_date,
    current_actual_upb,
    current_interest_bearing_upb,
    current_non_interest_bearing_upb,
    current_loan_delinquency_status,
    loan_age,
    current_interest_rate,
    NULLIF(eltv, 999) AS eltv,
    delinquent_accrued_interest,
    zero_balance_removal_upb,
    actual_loss,
    underwriting_defect_and_major_servicing_defect_settlement_date AS defect_settlement_date,
    modification_flag,
    mi_recoveries,
    net_sales_proceeds,
    non_mi_recoveries,
    total_expenses,
    maintenance_and_preservation_costs,
    taxes_and_insurance,
    miscellaneous_expenses,
    payment_deferral_flag,
    borrower_assistance_plan,
    bankruptcy_cramdown_costs
FROM raw.performance;

CREATE TABLE lgd.origination_data AS
SELECT 
    loan_identifier, 
    NULLIF(credit_score, 9999) AS credit_score,
    mortgage_insurance_pct,
    NULLIF(number_of_units, 99) AS number_of_units,
    occupancy_status,
    NULLIF(cltv, 999) AS cltv, 
    NULLIF(dti, 999) AS dti, 
    upb, 
    NULLIF(ltv, 999) AS ltv, 
    interest_rate,
    channel,
    property_state,
    loan_purpose,
    loan_term
FROM raw.origination;