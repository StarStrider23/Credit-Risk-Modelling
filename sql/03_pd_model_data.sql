DROP TABLE IF EXISTS pd.model_data;

CREATE TABLE pd.model_data AS
SELECT
    loan_identifier,
    observation_month,
    default_12m,

    NULLIF(credit_score, 9999) AS credit_score,
    first_time_homebuyer_indicator,
    mortgage_insurance_pct,
    NULLIF(number_of_units, 99) AS number_of_units,
    occupancy_status,
    NULLIF(cltv, 999) AS cltv,
    NULLIF(dti, 999) AS dti,
    original_upb,
    NULLIF(ltv, 999) AS ltv,
    interest_rate,
    channel,
    property_state,
    property_type,
    loan_purpose,
    loan_term,
    NULLIF(number_of_borrowers, 99) AS number_of_borrowers,
    current_actual_upb,
    loan_age,
    months_to_legal_maturity,
    current_interest_rate,
    current_non_interest_bearing_upb,
    NULLIF(eltv, 999) AS eltv,
    delinquency_status_numeric,
    modification_flag,
    payment_deferral_flag

FROM pd.features;