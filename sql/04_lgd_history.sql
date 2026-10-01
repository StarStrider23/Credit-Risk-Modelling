DROP TABLE IF EXISTS lgd.lgd_history;

CREATE TABLE lgd.lgd_history AS

WITH default_events_with_date AS (
    SELECT
        loan_identifier,
        observation_month AS default_month,
        TO_DATE(observation_month::text, 'YYYYMM') AS default_date
    FROM lgd.lgd_events
)

SELECT
    l.loan_identifier,
    l.observation_month,
    d.default_month,

    CASE
        WHEN l.observation_month = d.default_month
            THEN 'default_month'

        WHEN l.observation_month =
             TO_CHAR(d.default_date - INTERVAL '1 month', 'YYYYMM')::INTEGER
            THEN '1_month_before'

    END AS observation_type,

    l.current_actual_upb,
    l.current_loan_delinquency_status,
    l.loan_age,
    l.current_interest_rate,
    l.eltv,

    l.zero_balance_code,
    l.zero_balance_effective_date,
    l.delinquent_accrued_interest,
    l.zero_balance_removal_upb,
    l.actual_loss,

    l.modification_flag,
    l.payment_deferral_flag,
    l.borrower_assistance_plan

FROM lgd.lgd_data l

INNER JOIN default_events_with_date d
    ON l.loan_identifier = d.loan_identifier

WHERE l.observation_month IN (
    d.default_month,
    TO_CHAR(d.default_date - INTERVAL '1 month', 'YYYYMM')::INTEGER
);