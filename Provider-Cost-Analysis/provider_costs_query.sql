-- ============================================================================
-- SQL QUERY: Provider Costs Across All Currency Corridors
-- ============================================================================
-- This query captures provider costs across all currency corridors
-- Adjust table and column names based on your actual database schema
-- ============================================================================

-- Query 1: Comprehensive Provider Costs by Currency Corridor
-- This assumes a typical structure with providers, currencies, and cost data
SELECT
    p.provider_id,
    p.provider_name,
    pc.source_currency,
    pc.destination_currency,
    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
    pc.cost_amount,
    pc.cost_currency,
    pc.cost_type,  -- e.g., 'fixed', 'percentage', 'per_transaction'
    pc.effective_date,
    pc.expiry_date,
    CASE
        WHEN pc.expiry_date IS NULL OR pc.expiry_date > CURRENT_DATE
        THEN 'Active'
        ELSE 'Expired'
    END AS status,
    -- Calculate normalized cost if needed
    CASE
        WHEN pc.cost_type = 'percentage' THEN pc.cost_amount
        ELSE NULL
    END AS percentage_cost,
    CASE
        WHEN pc.cost_type = 'fixed' THEN pc.cost_amount
        ELSE NULL
    END AS fixed_cost,
    pc.created_at,
    pc.updated_at
FROM
    providers p
INNER JOIN
    provider_costs pc ON p.provider_id = pc.provider_id
WHERE
    pc.is_active = TRUE
ORDER BY
    p.provider_name,
    pc.source_currency,
    pc.destination_currency,
    pc.effective_date DESC;


-- ============================================================================
-- Query 2: Aggregated Provider Costs by Corridor
-- Useful for comparing costs across providers for the same corridor
-- ============================================================================
SELECT
    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
    pc.source_currency,
    pc.destination_currency,
    COUNT(DISTINCT p.provider_id) AS number_of_providers,
    AVG(CASE WHEN pc.cost_type = 'percentage' THEN pc.cost_amount END) AS avg_percentage_cost,
    MIN(CASE WHEN pc.cost_type = 'percentage' THEN pc.cost_amount END) AS min_percentage_cost,
    MAX(CASE WHEN pc.cost_type = 'percentage' THEN pc.cost_amount END) AS max_percentage_cost,
    AVG(CASE WHEN pc.cost_type = 'fixed' THEN pc.cost_amount END) AS avg_fixed_cost,
    MIN(CASE WHEN pc.cost_type = 'fixed' THEN pc.cost_amount END) AS min_fixed_cost,
    MAX(CASE WHEN pc.cost_type = 'fixed' THEN pc.cost_amount END) AS max_fixed_cost,
    STRING_AGG(DISTINCT p.provider_name, ', ') AS providers
FROM
    providers p
INNER JOIN
    provider_costs pc ON p.provider_id = pc.provider_id
WHERE
    pc.is_active = TRUE
    AND (pc.expiry_date IS NULL OR pc.expiry_date > CURRENT_DATE)
GROUP BY
    pc.source_currency,
    pc.destination_currency
ORDER BY
    pc.source_currency,
    pc.destination_currency;


-- ============================================================================
-- Query 3: Provider Cost Matrix - Wide Format
-- Pivot view for easy comparison across corridors
-- ============================================================================
SELECT
    p.provider_name,
    MAX(CASE WHEN CONCAT(pc.source_currency, '-', pc.destination_currency) = 'USD-EUR'
        THEN pc.cost_amount END) AS "USD_to_EUR",
    MAX(CASE WHEN CONCAT(pc.source_currency, '-', pc.destination_currency) = 'EUR-USD'
        THEN pc.cost_amount END) AS "EUR_to_USD",
    MAX(CASE WHEN CONCAT(pc.source_currency, '-', pc.destination_currency) = 'USD-GBP'
        THEN pc.cost_amount END) AS "USD_to_GBP",
    MAX(CASE WHEN CONCAT(pc.source_currency, '-', pc.destination_currency) = 'GBP-USD'
        THEN pc.cost_amount END) AS "GBP_to_USD",
    MAX(CASE WHEN CONCAT(pc.source_currency, '-', pc.destination_currency) = 'USD-NGN'
        THEN pc.cost_amount END) AS "USD_to_NGN",
    MAX(CASE WHEN CONCAT(pc.source_currency, '-', pc.destination_currency) = 'NGN-USD'
        THEN pc.cost_amount END) AS "NGN_to_USD"
    -- Add more currency corridors as needed
FROM
    providers p
INNER JOIN
    provider_costs pc ON p.provider_id = pc.provider_id
WHERE
    pc.is_active = TRUE
    AND (pc.expiry_date IS NULL OR pc.expiry_date > CURRENT_DATE)
GROUP BY
    p.provider_name
ORDER BY
    p.provider_name;


-- ============================================================================
-- Query 4: Dynamic Currency Corridors (All Combinations)
-- Lists all unique currency corridors available in the system
-- ============================================================================
SELECT DISTINCT
    pc.source_currency,
    pc.destination_currency,
    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
    COUNT(DISTINCT p.provider_id) AS provider_count,
    MIN(pc.effective_date) AS earliest_effective_date,
    MAX(pc.effective_date) AS latest_effective_date
FROM
    provider_costs pc
INNER JOIN
    providers p ON p.provider_id = pc.provider_id
WHERE
    pc.is_active = TRUE
GROUP BY
    pc.source_currency,
    pc.destination_currency
ORDER BY
    pc.source_currency,
    pc.destination_currency;


-- ============================================================================
-- Query 5: Provider Costs with Exchange Rate Context
-- Include exchange rates if you have a separate exchange rates table
-- ============================================================================
SELECT
    p.provider_id,
    p.provider_name,
    pc.source_currency,
    pc.destination_currency,
    CONCAT(pc.source_currency, ' -> ', pc.destination_currency) AS currency_corridor,
    pc.cost_amount,
    pc.cost_type,
    pc.cost_currency,
    er.exchange_rate,
    -- Calculate effective cost in destination currency
    CASE
        WHEN pc.cost_type = 'fixed' AND pc.cost_currency = pc.source_currency
        THEN pc.cost_amount * er.exchange_rate
        WHEN pc.cost_type = 'fixed' AND pc.cost_currency = pc.destination_currency
        THEN pc.cost_amount
        ELSE NULL
    END AS cost_in_destination_currency,
    pc.effective_date,
    er.rate_date
FROM
    providers p
INNER JOIN
    provider_costs pc ON p.provider_id = pc.provider_id
LEFT JOIN
    exchange_rates er ON er.source_currency = pc.source_currency
    AND er.destination_currency = pc.destination_currency
    AND er.rate_date = pc.effective_date
WHERE
    pc.is_active = TRUE
    AND (pc.expiry_date IS NULL OR pc.expiry_date > CURRENT_DATE)
ORDER BY
    p.provider_name,
    pc.source_currency,
    pc.destination_currency;


-- ============================================================================
-- SAMPLE SCHEMA (If you need to create tables)
-- ============================================================================
/*
CREATE TABLE providers (
    provider_id SERIAL PRIMARY KEY,
    provider_name VARCHAR(255) NOT NULL,
    provider_type VARCHAR(100),
    country VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE provider_costs (
    cost_id SERIAL PRIMARY KEY,
    provider_id INTEGER REFERENCES providers(provider_id),
    source_currency VARCHAR(3) NOT NULL,
    destination_currency VARCHAR(3) NOT NULL,
    cost_amount DECIMAL(10, 4) NOT NULL,
    cost_currency VARCHAR(3) NOT NULL,
    cost_type VARCHAR(50) NOT NULL,  -- 'fixed', 'percentage', 'per_transaction'
    effective_date DATE NOT NULL,
    expiry_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_provider_corridor UNIQUE (provider_id, source_currency, destination_currency, effective_date)
);

CREATE TABLE exchange_rates (
    rate_id SERIAL PRIMARY KEY,
    source_currency VARCHAR(3) NOT NULL,
    destination_currency VARCHAR(3) NOT NULL,
    exchange_rate DECIMAL(15, 6) NOT NULL,
    rate_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_currency_pair_date UNIQUE (source_currency, destination_currency, rate_date)
);

CREATE INDEX idx_provider_costs_corridor ON provider_costs(source_currency, destination_currency);
CREATE INDEX idx_provider_costs_provider ON provider_costs(provider_id);
CREATE INDEX idx_provider_costs_effective_date ON provider_costs(effective_date);
CREATE INDEX idx_exchange_rates_pair_date ON exchange_rates(source_currency, destination_currency, rate_date);
*/
