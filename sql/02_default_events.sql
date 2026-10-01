DROP TABLE IF EXISTS raw.default_events;

CREATE TABLE raw.default_events AS
SELECT
    loan_identifier,
    monthly_reporting_period AS default_date,
    zero_balance_code AS default_code
FROM raw.performance
WHERE zero_balance_code IN (2, 3, 9);