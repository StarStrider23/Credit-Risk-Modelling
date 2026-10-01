DROP TABLE IF EXISTS ecl.ecl;

CREATE TABLE ecl.ecl AS
SELECT
    loan_identifier,
    observation_month,
    NULLIF(credit_score, 9999) AS credit_score,
    NULLIF(mortgage_insurance_pct, 0) AS mortgage_insurance_pct,
    NULLIF(number_of_units, 9) AS number_of_units,
    NULLIF(cltv, 999) AS cltv,
    NULLIF(dti, 999) AS dti,
    original_upb,
    NULLIF(ltv, 999) AS ltv,
    interest_rate,
    loan_term,
    NULLIF(number_of_borrowers, 99) AS number_of_borrowers,
    NULLIF(first_time_homebuyer_indicator, '9') AS first_time_homebuyer_indicator,
    NULLIF(occupancy_status, '9') AS occupancy_status,
    NULLIF(channel, '9') AS channel,
    property_state,
    NULLIF(property_type, '99') AS property_type,
    NULLIF(loan_purpose, '9') AS loan_purpose,
    months_to_legal_maturity,
    current_actual_upb,
    current_interest_rate,
    current_non_interest_bearing_upb,
    NULLIF(eltv, 999) AS eltv,
    zero_balance_code,
    NULLIF(current_loan_delinquency_status, 'XX') AS current_loan_delinquency_status, 
    delinquency_status_numeric,
    modification_flag,
    payment_deferral_flag

FROM ecl.ingredients;