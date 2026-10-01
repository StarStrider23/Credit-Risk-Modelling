DROP TABLE IF EXISTS lgd.lgd_events;

CREATE TABLE lgd.lgd_events AS
SELECT *
FROM lgd.lgd_data
WHERE zero_balance_code IN (2, 3, 9);