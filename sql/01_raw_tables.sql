CREATE SCHEMA IF NOT EXISTS raw;

DROP TABLE IF EXISTS raw.origination;
DROP TABLE IF EXISTS raw.performance;

CREATE TABLE raw.origination (
    credit_score INTEGER,
    first_payment_date INTEGER,
    first_time_homebuyer_indicator TEXT,
    maturity_date INTEGER,
    msa_code TEXT,
    mortgage_insurance_pct INTEGER,
    number_of_units INTEGER,
    occupancy_status TEXT,
    cltv INTEGER, --original combined loan-to-value
    dti INTEGER, --original debt-to-income ratio
    upb INTEGER, --original unpaid principal balance
    ltv INTEGER, --original loan-to-value
    interest_rate NUMERIC,
    channel TEXT,
    prepayment_penalty_indicator CHAR(1),
    amortization_type VARCHAR(3),
    property_state VARCHAR(2),
    property_type TEXT,
    postal_code INTEGER,
    loan_identifier VARCHAR(12),
    loan_purpose TEXT,
    loan_term INTEGER,
    number_of_borrowers INTEGER,
    seller_name TEXT,
    super_conforming_flag CHAR(1),
    pre_harp_loan_sequence_number VARCHAR(12),
    special_eligibility_program TEXT,
    harp_indicator CHAR(1),
    property_valuation_method INTEGER,
    interest_only_indicator CHAR(1),
    vantage_score_4_0 INTEGER
);


CREATE TABLE raw.performance (
    loan_identifier VARCHAR(12),
    monthly_reporting_period INTEGER,
    current_actual_upb NUMERIC,
    current_loan_delinquency_status TEXT,
    loan_age INTEGER,
    months_to_legal_maturity INTEGER,
    underwriting_defect_and_major_servicing_defect_settlement_date INTEGER,
    modification_flag TEXT,
    zero_balance_code INTEGER,
    zero_balance_effective_date INTEGER,
    current_interest_rate NUMERIC,
    current_non_interest_bearing_upb NUMERIC,
    ddlpi INTEGER, --due date of last paid installement
    mi_recoveries NUMERIC,
    net_sales_proceeds NUMERIC,
    non_mi_recoveries NUMERIC,
    total_expenses NUMERIC,
    legal_costs NUMERIC,
    maintenance_and_preservation_costs NUMERIC,
    taxes_and_insurance NUMERIC,
    miscellaneous_expenses NUMERIC,
    actual_loss NUMERIC,
    cumulative_modification_costs NUMERIC,
    interest_rate_step_indicator TEXT,
    payment_deferral_flag CHAR(1),
    eltv INTEGER, --estimated loan-to-value
    zero_balance_removal_upb NUMERIC,
    delinquent_accrued_interest NUMERIC,
    delinquency_due_to_disaster TEXT,
    borrower_assistance_plan TEXT,
    current_period_modification_costs NUMERIC,
    current_interest_bearing_upb NUMERIC,
    mortgage_insurance_cancellation_indicator TEXT,
    servicer_name TEXT,
    bankruptcy_cramdown_costs NUMERIC
);