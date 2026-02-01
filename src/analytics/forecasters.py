"""
Price Forecasters for Agricultural Commodities
===============================================

Implements forecasting models:
- Moving Average
- ARIMA

Usage:
    from src.analytics.forecasters import ForecastEngine
    engine = ForecastEngine()
    report = engine.generate_report()
"""

import os
import sys
import json
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

# Suppress statsmodels warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# =============================================================================
# FORECASTERS
# =============================================================================
class MovingAverageForecaster:
    """Simple Moving Average forecaster."""
    
    def __init__(self, window: int = 7):
        self.window = window
        self.name = "Moving Average"
    
    def forecast(self, series: pd.Series, periods: int = 7) -> List[float]:
        """Generate forecast using moving average trend."""
        if len(series) < self.window:
            return [series.mean()] * periods
        
        ma = series.tail(self.window).mean()
        trend = (series.iloc[-1] - series.iloc[-self.window]) / self.window
        
        forecasts = []
        for i in range(1, periods + 1):
            forecasts.append(round(ma + trend * i, 2))
        
        return forecasts


class ARIMAForecaster:
    """ARIMA time series forecaster."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        self.order = order
        self.name = "ARIMA"
    
    def forecast(self, series: pd.Series, periods: int = 7) -> List[float]:
        """Generate ARIMA forecast."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            if len(series) < 30:
                return [series.mean()] * periods
            
            model = ARIMA(series, order=self.order)
            fitted = model.fit()
            forecasts = fitted.forecast(steps=periods)
            
            return [round(f, 2) for f in forecasts.tolist()]
            
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}, using fallback")
            return [round(series.mean(), 2)] * periods


# =============================================================================
# EVALUATION METRICS
# =============================================================================
def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mask = actual != 0
    if not mask.any():
        return 0.0
    
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return round(mape, 2)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return round(rmse, 2)


# =============================================================================
# FORECAST ENGINE
# =============================================================================
class ForecastEngine:
    """
    Forecasting engine for agricultural commodities.
    
    Runs multiple models and evaluates performance.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the forecast engine.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or config.DATABASE_PATH
        self.forecasters = {
            'moving_average': MovingAverageForecaster(),
            'arima': ARIMAForecaster()
        }
        self.results: Dict[str, Any] = {}
    
    def get_price_series(
        self,
        commodity: str,
        days: int = 365
    ) -> pd.Series:
        """Get price time series for a commodity."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT p.date as date,
                   AVG(p.price) as price
            FROM price_data p
            JOIN commodities c ON p.commodity_id = c.commodity_id
            WHERE c.name = ?
              AND p.date >= date('now', ?)
            GROUP BY p.date
            ORDER BY p.date
        """
        
        df = pd.read_sql_query(query, conn, params=(commodity, f'-{days} days'))
        conn.close()
        
        if df.empty:
            return pd.Series(dtype=float)
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df['price']
    
    def evaluate_model(
        self,
        series: pd.Series,
        forecaster: Any,
        test_size: int = 30
    ) -> Dict[str, float]:
        """
        Evaluate forecaster on historical data.
        
        Args:
            series: Price series
            forecaster: Forecaster instance
            test_size: Number of points to use for testing
            
        Returns:
            Evaluation metrics
        """
        if len(series) < test_size + 30:
            return {'mape': 0.0, 'rmse': 0.0}
        
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        
        # Generate predictions for each test point
        predictions = []
        for i in range(len(test)):
            train_subset = series.iloc[:-(test_size - i)]
            pred = forecaster.forecast(train_subset, periods=1)
            predictions.append(pred[0])
        
        mape = calculate_mape(test.values, np.array(predictions))
        rmse = calculate_rmse(test.values, np.array(predictions))
        
        return {'mape': mape, 'rmse': rmse}
    
    def forecast_commodity(
        self,
        commodity: str,
        forecast_days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Generate forecasts for a commodity.
        
        Args:
            commodity: Commodity name
            forecast_days: Days to forecast
            
        Returns:
            Forecast results
        """
        logger.info(f"Forecasting {commodity} for {forecast_days} days...")
        
        series = self.get_price_series(commodity)
        
        if series.empty or len(series) < 30:
            logger.warning(f"No data for {commodity}")
            return None
        
        current_price = series.iloc[-1]
        model_results = {}
        best_model = None
        best_mape = float('inf')
        
        for name, forecaster in self.forecasters.items():
            # Generate forecast
            forecast = forecaster.forecast(series, forecast_days)
            
            # Evaluate
            metrics = self.evaluate_model(series, forecaster)
            
            model_results[name] = {
                'forecast': forecast,
                'mape': metrics['mape'],
                'rmse': metrics['rmse']
            }
            
            if metrics['mape'] < best_mape:
                best_mape = metrics['mape']
                best_model = name
        
        result = {
            'commodity': commodity,
            'current_price': round(current_price, 2),
            'forecast_days': forecast_days,
            'forecast_7day': model_results.get(best_model, {}).get('forecast', []),
            'best_model': self.forecasters[best_model].name if best_model else 'Unknown',
            'model_accuracy': {
                'mape': model_results.get(best_model, {}).get('mape', 0),
                'rmse': model_results.get(best_model, {}).get('rmse', 0)
            },
            'models': model_results
        }
        
        self.results[commodity] = result
        return result
    
    def forecast_all(
        self,
        commodities: List[str] = None,
        forecast_days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate forecasts for all commodities.
        
        Args:
            commodities: List of commodities (None = all)
            forecast_days: Days to forecast
            
        Returns:
            All forecast results
        """
        if commodities is None:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT DISTINCT name FROM commodities")
            commodities = [row[0] for row in cursor.fetchall()]
            conn.close()
        
        logger.info(f"Forecasting {len(commodities)} commodities...")
        
        for commodity in commodities:
            self.forecast_commodity(commodity, forecast_days)
        
        return self.results
    
    def generate_report(
        self,
        output_file: str = None,
        commodities: List[str] = None,
        forecast_days: int = 7
    ) -> str:
        """
        Generate forecast report.
        
        Args:
            output_file: Output JSON file path
            commodities: Commodities to forecast
            forecast_days: Days to forecast
            
        Returns:
            Path to generated report
        """
        if output_file is None:
            output_file = str(config.REPORTS_DIR / "forecast_output.json")
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.forecast_all(commodities, forecast_days)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'forecast_period_days': forecast_days,
            'total_commodities': len(self.results),
            'forecasts': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {output_file}")
        return output_file


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Price Forecaster')
    parser.add_argument('--db', type=str, default=None, help='Database path')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--days', type=int, default=7, help='Forecast days')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Agricultural Commodity Price Forecaster")
    print("=" * 50)
    
    engine = ForecastEngine(db_path=args.db)
    report_path = engine.generate_report(args.output, forecast_days=args.days)
    
    print(f"\nForecasted {len(engine.results)} commodities")
    print(f"Report saved to: {report_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
