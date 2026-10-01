DROP TABLE IF EXISTS pd.features;

CREATE TABLE pd.features AS
SELECT
    loan_identifier,
    observation_month,

    -- Target
    default_12m,

    -- Origination characteristics
    credit_score,
    first_time_homebuyer_indicator,
    mortgage_insurance_pct,
    number_of_units,
    occupancy_status,
    cltv,
    dti,
    original_upb,
    ltv,
    interest_rate,
    channel,
    property_state,
    property_type,
    loan_purpose,
    loan_term,
    number_of_borrowers

    -- Current loan characteristics
    current_actual_upb,
    loan_age,
    months_to_legal_maturity,
    current_interest_rate,
    current_non_interest_bearing_upb,
    eltv,
    modification_flag,
    payment_deferral_flag,

    -- Delinquency
    CASE
        WHEN current_loan_delinquency_status ~ '^[0-9]+$'
        THEN current_loan_delinquency_status::INTEGER
        ELSE NULL
    END AS delinquency_status_numeric

FROM pd.dataset;