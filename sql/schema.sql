-- =============================================================================
-- INNOFARMS Agricultural Market Intelligence Database Schema
-- =============================================================================

-- Drop existing tables if they exist (for clean setup)

DROP TABLE IF EXISTS daily_aggregates;

DROP TABLE IF EXISTS price_data;

DROP TABLE IF EXISTS sources;

DROP TABLE IF EXISTS commodities;

-- =============================================================================
-- Table: commodities
-- Stores master list of agricultural commodities
-- =============================================================================
CREATE TABLE commodities (
    commodity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50) NOT NULL
);

-- =============================================================================
-- Table: sources
-- Stores data source information
-- =============================================================================
CREATE TABLE sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    url VARCHAR(500),
    reliability_score DECIMAL(3, 1) CHECK (
        reliability_score >= 0
        AND reliability_score <= 10
    )
);

-- =============================================================================
-- Table: price_data
-- Stores raw price data from all sources
-- =============================================================================
-- =============================================================================
-- Table: price_data
-- Stores raw price data from all sources
-- =============================================================================
CREATE TABLE price_data (
    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    unit VARCHAR(50),
    date DATE NOT NULL,
    FOREIGN KEY (commodity_id) REFERENCES commodities (commodity_id),
    FOREIGN KEY (source_id) REFERENCES sources (source_id)
);

-- =============================================================================
-- Table: daily_aggregates
-- Stores daily aggregated statistics per commodity
-- =============================================================================
CREATE TABLE daily_aggregates (
    commodity_id INTEGER NOT NULL,
    date DATE NOT NULL,
    avg_price DECIMAL(10, 2),
    min DECIMAL(10, 2),
    max DECIMAL(10, 2),
    std_dev DECIMAL(10, 4),
    FOREIGN KEY (commodity_id) REFERENCES commodities (commodity_id),
    -- Unique constraint for simple lookup, though not explicitly asked, is standard for 'aggregates'
    UNIQUE (commodity_id, date)
);

-- =============================================================================
-- Indexes for better query performance
-- =============================================================================
CREATE INDEX idx_price_data_commodity ON price_data (commodity_id);

CREATE INDEX idx_price_data_date ON price_data (date);

CREATE INDEX idx_price_data_source ON price_data (source_id);

CREATE INDEX idx_daily_aggregates_commodity ON daily_aggregates (commodity_id);

CREATE INDEX idx_daily_aggregates_date ON daily_aggregates (date);

-- =============================================================================
-- Insert initial data: Commodities
-- =============================================================================
INSERT INTO
    commodities (name, category)
VALUES ('Wheat', 'Cereals'),
    ('Rice', 'Cereals'),
    ('Corn', 'Cereals'), -- Used 'Maize' before, standardizing to 'Corn' to match Investing/USDA commonly, but Agmarknet uses Maize. Will use Corn as primary name.
    ('Tomato', 'Vegetables'),
    ('Onion', 'Vegetables');

-- =============================================================================
-- Insert initial data: Sources
-- =============================================================================
INSERT INTO
    sources (name, url, reliability_score)
VALUES (
        'AGMARKNET',
        'https://agmarknet.gov.in',
        9.0
    ),
    (
        'USDA NASS',
        'https://quickstats.nass.usda.gov',
        10.0
    ),
    (
        'Investing.com',
        'https://www.investing.com/commodities/agricultural',
        8.0
    );