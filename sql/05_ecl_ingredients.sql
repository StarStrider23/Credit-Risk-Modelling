CREATE SCHEMA IF NOT EXISTS ecl;

DROP TABLE IF EXISTS ecl.ingredients;

CREATE TABLE ecl.ingredients AS
SELECT
    p.loan_identifier,
    p.monthly_reporting_period AS observation_month,
    
    o.credit_score,
    o.mortgage_insurance_pct,
    o.number_of_units,
    o.cltv,
    o.dti,
    o.upb AS original_upb,
    o.ltv,
    o.interest_rate,
    o.loan_term,
    o.number_of_borrowers,
    o.first_time_homebuyer_indicator,
    o.occupancy_status,
    o.channel,
    o.property_state,
    o.property_type,
    o.loan_purpose,

    p.loan_age,
    p.current_actual_upb,
    p.current_interest_rate,
    p.current_non_interest_bearing_upb,
    p.eltv,
    p.zero_balance_code,
    p.current_loan_delinquency_status,
    p.modification_flag,
    p.payment_deferral_flag,
    CASE
        WHEN p.current_loan_delinquency_status ~ '^[0-9]+$'
        THEN p.current_loan_delinquency_status::INTEGER
        ELSE NULL
    END AS delinquency_status_numeric

FROM raw.performance p

JOIN raw.origination o
    ON p.loan_identifier = o.loan_identifier

WHERE p.monthly_reporting_period = '202603';
