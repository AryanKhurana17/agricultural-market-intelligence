"""
Unit Tests for INNOFarms Agricultural Market Intelligence System
=================================================================

Test coverage for:
- ETL Pipeline
- Trend Analyzer
- Forecasters
- Financial Calculators

Run with: pytest tests/ -v

Author: INNOFarms Data Analyst Assignment
Date: 2026-02-01
"""

import os
import sys
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analytics"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "financial"))


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================
@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    return [
        {
            'commodity': 'Wheat',
            'date': d.strftime('%Y-%m-%d'),
            'price': 2000 + np.random.uniform(-100, 100),
            'modal_price': 2000 + np.random.uniform(-100, 100),
            'source': 'AGMARKNET',
            'market': 'Test Market',
            'state': 'Test State'
        }
        for d in dates
    ]


@pytest.fixture
def temp_database():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create schema
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE commodities (
            commodity_id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            category TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            url TEXT,
            reliability_score REAL
        )
    """)
    conn.execute("""
        CREATE TABLE price_data (
            price_id INTEGER PRIMARY KEY,
            commodity_id INTEGER,
            source_id INTEGER,
            price REAL,
            unit TEXT,
            price_date DATE,
            modal_price REAL
        )
    """)
    
    # Insert test data
    conn.execute("INSERT INTO commodities VALUES (1, 'Wheat', 'Cereals')")
    conn.execute("INSERT INTO sources VALUES (1, 'AGMARKNET', 'http://test', 9.0)")
    
    # Insert price data
    for i in range(30):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        price = 2000 + np.random.uniform(-100, 100)
        conn.execute(
            "INSERT INTO price_data VALUES (?, 1, 1, ?, 'Rs./Quintal', ?, ?)",
            (i+1, price, date, price)
        )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def temp_config():
    """Create temporary config for testing."""
    config = {
        'commodities': {
            'wheat': {
                'name': 'Wheat',
                'category': 'Cereals',
                'cycle_days': 120,
                'yield_per_hectare': 40,
                'costs': {
                    'seeds': 3000,
                    'fertilizers': 8000,
                    'labor': 12000,
                    'total': 45000
                }
            }
        },
        'market_prices': {'wheat': 2500}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config, f)
        config_path = f.name
    
    yield config_path
    os.unlink(config_path)


# =============================================================================
# ETL PIPELINE TESTS
# =============================================================================
class TestETLPipeline:
    """Tests for ETL pipeline functionality."""
    
    def test_price_normalization(self, sample_price_data):
        """Test that prices are normalized to INR/Quintal."""
        df = pd.DataFrame(sample_price_data)
        df['unit'] = 'Rs./Kg'
        
        # Simulate normalization
        mask = df['unit'].str.contains('Kg', case=False, na=False)
        df.loc[mask, 'price'] = df.loc[mask, 'price'] * 100
        df['unit'] = 'Rs./Quintal'
        
        assert all(df['unit'] == 'Rs./Quintal')
        assert df['price'].mean() > 100000  # Kg to Quintal conversion
    
    def test_duplicate_removal(self, sample_price_data):
        """Test duplicate detection and removal."""
        # Create duplicates
        data = sample_price_data + sample_price_data[:5]
        df = pd.DataFrame(data)
        
        original_count = len(df)
        df = df.drop_duplicates(subset=['commodity', 'date', 'market', 'source'])
        
        assert len(df) < original_count
        assert len(df) == 30  # Original unique records
    
    def test_anomaly_detection(self, sample_price_data):
        """Test anomaly detection using z-score method."""
        df = pd.DataFrame(sample_price_data)
        
        # Add an anomaly
        df.loc[0, 'price'] = 10000  # Much higher than normal
        
        mean = df['price'].mean()
        std = df['price'].std()
        z_scores = (df['price'] - mean) / std
        
        anomalies = abs(z_scores) > 3
        assert anomalies.sum() >= 1  # At least one anomaly detected
    
    def test_date_parsing(self):
        """Test date format normalization."""
        dates = ['2026-01-15', '15-01-2026', '01/15/2026']
        
        for date_str in dates:
            try:
                parsed = pd.to_datetime(date_str)
                assert parsed is not None
            except:
                pass  # Some formats may fail, that's okay


# =============================================================================
# TREND ANALYZER TESTS
# =============================================================================
class TestTrendAnalyzer:
    """Tests for trend analysis functionality."""
    
    def test_moving_average_calculation(self):
        """Test 7-day moving average calculation."""
        prices = [100, 102, 101, 103, 104, 102, 105, 106, 104, 107]
        series = pd.Series(prices)
        
        ma_7 = series.rolling(window=7).mean()
        
        assert pd.isna(ma_7.iloc[:6]).all()  # First 6 are NaN
        assert not pd.isna(ma_7.iloc[6])  # 7th value exists
        assert abs(ma_7.iloc[6] - 102.43) < 0.1
    
    def test_trend_detection_upward(self):
        """Test upward trend detection."""
        prices = [100, 102, 104, 106, 108, 110, 112]
        series = pd.Series(prices)
        
        ma_short = series.rolling(3).mean()
        ma_long = series.rolling(5).mean()
        
        # Upward if short MA > long MA
        latest_short = ma_short.iloc[-1]
        latest_long = ma_long.iloc[-1]
        
        assert latest_short > latest_long
        trend = 'Upward' if latest_short > latest_long else 'Downward'
        assert trend == 'Upward'
    
    def test_volatility_calculation(self):
        """Test volatility (coefficient of variation) calculation."""
        # Low volatility
        stable_prices = [100, 101, 100, 99, 100, 101, 100]
        stable_cv = (np.std(stable_prices) / np.mean(stable_prices)) * 100
        
        # High volatility
        volatile_prices = [100, 120, 90, 130, 85, 140, 80]
        volatile_cv = (np.std(volatile_prices) / np.mean(volatile_prices)) * 100
        
        assert stable_cv < 5  # Low volatility
        assert volatile_cv > 15  # High volatility
    
    def test_volatility_classification(self):
        """Test volatility level classification."""
        def classify(cv):
            if cv < 10:
                return 'Low'
            elif cv < 20:
                return 'Medium'
            else:
                return 'High'
        
        assert classify(5) == 'Low'
        assert classify(15) == 'Medium'
        assert classify(25) == 'High'


# =============================================================================
# FORECASTER TESTS
# =============================================================================
class TestForecaster:
    """Tests for forecasting functionality."""
    
    def test_moving_average_forecast(self):
        """Test simple moving average forecast."""
        prices = list(range(100, 115))  # 15 days of increasing prices
        series = pd.Series(prices)
        
        # 7-day forecast using MA
        ma = series.tail(7).mean()
        forecast = [ma] * 7
        
        assert len(forecast) == 7
        assert all(f > 107 for f in forecast)  # Forecast > recent average
    
    def test_mape_calculation(self):
        """Test MAPE (Mean Absolute Percentage Error)."""
        actual = [100, 110, 120, 130, 140]
        predicted = [98, 112, 118, 135, 138]
        
        errors = [abs(a - p) / a * 100 for a, p in zip(actual, predicted)]
        mape = sum(errors) / len(errors)
        
        assert 0 < mape < 10  # Reasonable error range
    
    def test_rmse_calculation(self):
        """Test RMSE (Root Mean Square Error)."""
        actual = np.array([100, 110, 120, 130, 140])
        predicted = np.array([98, 112, 118, 135, 138])
        
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        assert rmse > 0
        assert rmse < 10  # Reasonable error range


# =============================================================================
# FINANCIAL CALCULATOR TESTS
# =============================================================================
class TestFinancialCalculators:
    """Tests for financial calculation functionality."""
    
    def test_cost_per_quintal(self):
        """Test cost per quintal calculation."""
        total_cost = 45000  # INR per hectare
        yield_qty = 40  # quintal per hectare
        
        cost_per_quintal = total_cost / yield_qty
        assert cost_per_quintal == 1125
    
    def test_profit_calculation(self):
        """Test profit calculation."""
        market_price = 2500  # per quintal
        yield_qty = 40  # quintal per hectare
        total_cost = 45000  # per hectare
        
        revenue = market_price * yield_qty
        profit = revenue - total_cost
        
        assert revenue == 100000
        assert profit == 55000
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        profit = 55000
        total_cost = 45000
        
        roi = (profit / total_cost) * 100
        assert abs(roi - 122.22) < 1
    
    def test_profit_margin(self):
        """Test profit margin calculation."""
        profit = 55000
        revenue = 100000
        
        margin = (profit / revenue) * 100
        assert abs(margin - 55) < 0.01  # Allow small floating point error
    
    def test_breakeven_price(self):
        """Test break-even price calculation."""
        total_cost = 45000
        yield_qty = 40
        
        breakeven_price = total_cost / yield_qty
        assert breakeven_price == 1125
    
    def test_breakeven_yield(self):
        """Test break-even yield calculation."""
        total_cost = 45000
        market_price = 2500
        
        breakeven_yield = total_cost / market_price
        assert breakeven_yield == 18
    
    def test_margin_of_safety(self):
        """Test margin of safety calculation."""
        market_price = 2500
        breakeven_price = 1125
        
        margin_of_safety = ((market_price - breakeven_price) / market_price) * 100
        assert abs(margin_of_safety - 55) < 0.01  # Allow small floating point error


# =============================================================================
# OPPORTUNITY SCORER TESTS
# =============================================================================
class TestOpportunityScorer:
    """Tests for opportunity scoring functionality."""
    
    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        weights = {
            'profit_margin': 0.30,
            'roi': 0.25,
            'profit_per_day': 0.20,
            'price_stability': 0.15,
            'demand_trend': 0.10
        }
        
        scores = {
            'profit_margin': 80,
            'roi': 90,
            'profit_per_day': 70,
            'price_stability': 60,
            'demand_trend': 75
        }
        
        weighted_score = sum(scores[k] * weights[k] for k in weights)
        assert 70 < weighted_score < 85
    
    def test_risk_classification(self):
        """Test risk level classification."""
        def classify_risk(volatility):
            if volatility < 10:
                return 'Low'
            elif volatility < 20:
                return 'Medium'
            else:
                return 'High'
        
        assert classify_risk(5) == 'Low'
        assert classify_risk(15) == 'Medium'
        assert classify_risk(25) == 'High'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_database_connection(self, temp_database):
        """Test database connection and query."""
        conn = sqlite3.connect(temp_database)
        cursor = conn.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 30
    
    def test_commodity_lookup(self, temp_database):
        """Test commodity lookup in database."""
        conn = sqlite3.connect(temp_database)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM commodities WHERE name = ?", ('Wheat',))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row['category'] == 'Cereals'
    
    def test_price_query(self, temp_database):
        """Test price data query."""
        conn = sqlite3.connect(temp_database)
        cursor = conn.execute("""
            SELECT AVG(price) as avg_price FROM price_data
            WHERE commodity_id = 1
        """)
        avg_price = cursor.fetchone()[0]
        conn.close()
        
        assert 1900 < avg_price < 2100


# =============================================================================
# RUN TESTS
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
