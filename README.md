# INNOFarms - Agricultural Market Intelligence System

A comprehensive agricultural commodity market analysis platform with data scraping, ETL processing, trend analysis, price forecasting, and financial modeling.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your USDA API key

# 3. Run scrapers
python src/scrapers/main_scraper.py

# 4. Run ETL pipeline
python src/etl/etl_pipeline.py

# 5. Run analysis
python src/main_analysis.py
```

## Project Structure

```
INNOFARMS/
├── config/                    # Configuration files
│   ├── commodities.yaml       # Commodity definitions
│   └── production_costs.yaml  # Cost parameters
├── data/raw/                  # Raw scraped data (JSON)
├── database/                  # SQLite database
├── logs/                      # Application logs
├── reports/                   # Generated reports
│   └── visualizations/        # Generated charts
├── sql/                       # Database schema
├── src/
│   ├── utils/                 # Shared utilities
│   │   ├── logger.py          # Centralized logging
│   │   └── config.py          # Environment configuration
│   ├── scrapers/              # Data scrapers
│   │   ├── main_scraper.py    # Scraper orchestrator
│   │   ├── usda_scraper.py    # USDA NASS API
│   │   ├── investing_scraper.py
│   │   └── agmarknet_scraper.py
│   ├── etl/
│   │   └── etl_pipeline.py    # Extract-Transform-Load
│   ├── analytics/
│   │   ├── trend_analyzer.py  # Moving averages, volatility
│   │   └── forecasters.py     # ARIMA, MA forecasting
│   ├── financial/
│   │   ├── cost_calculator.py
│   │   ├── profit_calculator.py
│   │   ├── opportunity_scorer.py
│   │   ├── risk_analyzer.py
│   │   └── scenario_planner.py
│   ├── visualizations/
│   │   └── visualizer.py      # Chart generation
│   └── main_analysis.py       # Analysis runner
├── tests/
│   └── test_pipeline.py       # Unit tests (23 tests)
├── .env                       # Environment variables (API keys)
├── .env.example               # Environment template
└── requirements.txt           # Python dependencies
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
USDA_API_KEY=your_api_key_here  # Get from quickstats.nass.usda.gov/api
DATABASE_PATH=database/market_data.db
LOG_LEVEL=INFO
```

## Commands

### Scrapers

```bash
# Run all scrapers
python src/scrapers/main_scraper.py

# Run specific sources
python src/scrapers/main_scraper.py --sources usda,investing

# Run with options
python src/scrapers/main_scraper.py --sources usda --years 5
```

### ETL Pipeline

```bash
# Run ETL
python src/etl/etl_pipeline.py

# Specify input/output
python src/etl/etl_pipeline.py --input data/raw --db database/market_data.db
```

### Analysis

```bash
# Run all analysis
python src/main_analysis.py

# Generate visualizations
python src/visualizations/visualizer.py
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

## Modularity

### Adding New Commodities

Edit `config/commodities.yaml` to add new commodities. Each scraper has extendable commodity configurations:

```python
# In scraper file
from src.scrapers.usda_scraper import USDANASSScraper

scraper = USDANASSScraper()
scraper.add_commodity("BARLEY", {
    "commodity_desc": "BARLEY",
    "statisticcat_desc": "PRICE RECEIVED",
    "unit_desc": "$ / BU"
})
```

### Adding New Sources

Create a new scraper in `src/scrapers/` following the base pattern:

1. Create `your_scraper.py`
2. Implement `scrape_commodity()` and `scrape_all_commodities()` methods
3. Register in `main_scraper.py`

## Data Sources

| Source | Type | Data | Coverage |
|--------|------|------|----------|
| USDA NASS | API | US prices | Historical |
| Investing.com | Sample | Futures | 10 years |
| AGMARKNET | Sample | Indian APMC | 7 days |

## Generated Reports

| Report | Description |
|--------|-------------|
| `trend_analysis.json` | Moving averages, volatility |
| `forecast_output.json` | 7-day price predictions |
| `cost_analysis.json` | Production cost breakdown |
| `profitability_report.json` | ROI and margins |
| `opportunity_rankings.json` | Investment scores |
| `risk_analysis.json` | Risk assessment |
| `scenario_analysis.json` | What-if scenarios |

## Technology Stack

- **Language**: Python 3.10+
- **Database**: SQLite
- **Analytics**: pandas, numpy, statsmodels
- **Visualization**: matplotlib
- **Testing**: pytest

## License

MIT License
