CREATE SCHEMA IF NOT EXISTS pd;

DROP TABLE IF EXISTS pd.dataset;

CREATE TABLE pd.dataset AS
SELECT
    p.loan_identifier,
    p.monthly_reporting_period AS observation_month,

    -- Origination characteristics
    o.credit_score,
    o.first_payment_date,
    o.first_time_homebuyer_indicator,
    o.maturity_date,
    o.mortgage_insurance_pct,
    o.number_of_units,
    o.occupancy_status,
    o.cltv,
    o.dti,
    o.upb AS original_upb,
    o.ltv,
    o.interest_rate,
    o.channel,
    o.prepayment_penalty_indicator,
    o.amortization_type,
    o.property_state,
    o.property_type,
    o.postal_code,
    o.loan_purpose,
    o.loan_term,
    o.number_of_borrowers

    -- Current performance
    p.current_actual_upb,
    p.current_loan_delinquency_status,
    p.loan_age,
    p.months_to_legal_maturity,
    p.current_interest_rate,
    p.current_non_interest_bearing_upb,
    p.eltv,
    p.borrower_assistance_plan,
    p.modification_flag,
    p.payment_deferral_flag,

    -- Target
CASE
    WHEN d.default_date IS NOT NULL
     AND TO_DATE(d.default_date::TEXT, 'YYYYMM')
         > TO_DATE(p.monthly_reporting_period::TEXT, '*YYYYMM')
     AND TO_DATE(d.default_date::TEXT, 'YYYYMM')
         <= TO_DATE(p.monthly_reporting_period::TEXT, 'YYYYMM')
            + INTERVAL '12 months'
    THEN 1
    ELSE 0
END AS default_12m

FROM raw.performance p

JOIN raw.origination o
    ON p.loan_identifier = o.loan_identifier

LEFT JOIN raw.default_events d
    ON p.loan_identifier = d.loan_identifier

-- Need a complete 12-month observation window
WHERE TO_DATE(p.monthly_reporting_period::TEXT, 'YYYYMM')
      <= DATE '2025-03-01'
  AND (
      d.default_date IS NULL
      OR p.monthly_reporting_period < d.default_date
  );