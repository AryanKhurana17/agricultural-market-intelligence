"""
Trend Analyzer for Agricultural Commodities
============================================

Calculates moving averages, trend detection, and volatility metrics.

Usage:
    from src.analytics.trend_analyzer import TrendAnalyzer
    analyzer = TrendAnalyzer()
    report = analyzer.generate_report()
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# =============================================================================
# TREND ANALYZER CLASS
# =============================================================================
class TrendAnalyzer:
    """
    Analyzes price trends for agricultural commodities.
    
    Features:
    - Moving averages (7, 15, 30 day)
    - Trend detection (upward/downward/stable)
    - Volatility calculation (coefficient of variation)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the trend analyzer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or config.DATABASE_PATH
        self.results: Dict[str, Any] = {}
    
    def get_price_series(
        self,
        commodity: str,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Get price time series for a commodity.
        
        Args:
            commodity: Commodity name
            days: Number of days of history
            
        Returns:
            DataFrame with date and price columns
        """
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
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def calculate_moving_averages(
        self,
        df: pd.DataFrame,
        windows: List[int] = [7, 15, 30]
    ) -> Dict[str, float]:
        """
        Calculate moving averages for different windows.
        
        Args:
            df: Price DataFrame
            windows: List of window sizes
            
        Returns:
            Dict of moving average values
        """
        result = {}
        
        for window in windows:
            if len(df) >= window:
                ma = df['price'].rolling(window=window).mean().iloc[-1]
                result[f'ma_{window}_day'] = round(ma, 2)
            else:
                result[f'ma_{window}_day'] = None
        
        return result
    
    def detect_trend(
        self,
        df: pd.DataFrame,
        short_window: int = 7,
        long_window: int = 30
    ) -> Dict[str, Any]:
        """
        Detect price trend direction.
        
        Args:
            df: Price DataFrame
            short_window: Short-term MA window
            long_window: Long-term MA window
            
        Returns:
            Trend analysis result
        """
        if len(df) < long_window:
            return {'trend': 'Unknown', 'strength': 0}
        
        short_ma = df['price'].rolling(window=short_window).mean().iloc[-1]
        long_ma = df['price'].rolling(window=long_window).mean().iloc[-1]
        
        if long_ma == 0:
            return {'trend': 'Unknown', 'strength': 0}
        
        diff_pct = ((short_ma - long_ma) / long_ma) * 100
        
        if diff_pct > 2:
            trend = 'Upward'
        elif diff_pct < -2:
            trend = 'Downward'
        else:
            trend = 'Stable'
        
        return {
            'trend': trend,
            'strength': round(abs(diff_pct), 2)
        }
    
    def calculate_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate price volatility using coefficient of variation.
        
        Args:
            df: Price DataFrame
            
        Returns:
            Volatility metrics
        """
        if df.empty or len(df) < 10:
            return {'volatility_pct': 0, 'level': 'Unknown'}
        
        mean_price = df['price'].mean()
        std_price = df['price'].std()
        
        if mean_price == 0:
            return {'volatility_pct': 0, 'level': 'Unknown'}
        
        cv = (std_price / mean_price) * 100
        
        if cv < 10:
            level = 'Low'
        elif cv < 20:
            level = 'Medium'
        else:
            level = 'High'
        
        return {
            'volatility_pct': round(cv, 2),
            'level': level
        }
    
    def analyze_commodity(
        self,
        commodity: str,
        days: int = 365
    ) -> Optional[Dict[str, Any]]:
        """
        Perform complete analysis for a commodity.
        
        Args:
            commodity: Commodity name
            days: Days of historical data
            
        Returns:
            Complete analysis result
        """
        logger.info(f"Analyzing {commodity}...")
        
        df = self.get_price_series(commodity, days)
        
        if df.empty or len(df) < 7:
            logger.warning(f"No data for {commodity}")
            return None
        
        # Current and stats
        current_price = df['price'].iloc[-1]
        stats = {
            'min': round(df['price'].min(), 2),
            'max': round(df['price'].max(), 2),
            'average': round(df['price'].mean(), 2),
            'std_dev': round(df['price'].std(), 2)
        }
        
        # Moving averages
        mas = self.calculate_moving_averages(df)
        
        # Trend
        trend = self.detect_trend(df)
        
        # Volatility
        volatility = self.calculate_volatility(df)
        
        result = {
            'commodity': commodity,
            'current_price': round(current_price, 2),
            '7day_avg': mas.get('ma_7_day'),
            '15day_avg': mas.get('ma_15_day'),
            '30day_avg': mas.get('ma_30_day'),
            'trend': trend['trend'],
            'trend_strength': trend['strength'],
            'volatility': f"{volatility['level']} ({volatility['volatility_pct']}%)",
            'volatility_pct': volatility['volatility_pct'],
            'volatility_level': volatility['level'],
            'statistics': stats,
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'data_points': len(df)
            }
        }
        
        self.results[commodity] = result
        return result
    
    def analyze_all(
        self,
        commodities: List[str] = None,
        days: int = 365
    ) -> Dict[str, Any]:
        """
        Analyze all commodities.
        
        Args:
            commodities: List of commodities (None = all in DB)
            days: Days of historical data
            
        Returns:
            Analysis results for all commodities
        """
        if commodities is None:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT DISTINCT name FROM commodities")
            commodities = [row[0] for row in cursor.fetchall()]
            conn.close()
        
        logger.info(f"Analyzing {len(commodities)} commodities...")
        
        for commodity in commodities:
            self.analyze_commodity(commodity, days)
        
        return self.results
    
    def generate_report(
        self,
        output_file: str = None,
        commodities: List[str] = None
    ) -> str:
        """
        Generate trend analysis report.
        
        Args:
            output_file: Output JSON file path
            commodities: Commodities to analyze
            
        Returns:
            Path to generated report
        """
        if output_file is None:
            output_file = str(config.REPORTS_DIR / "trend_analysis.json")
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.analyze_all(commodities)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'analysis_period_days': 365,
            'total_commodities': len(self.results),
            'commodities': self.results
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
    
    parser = argparse.ArgumentParser(description='Trend Analyzer')
    parser.add_argument('--db', type=str, default=None, help='Database path')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--days', type=int, default=365, help='Analysis period')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Agricultural Commodity Trend Analyzer")
    print("=" * 50)
    
    analyzer = TrendAnalyzer(db_path=args.db)
    report_path = analyzer.generate_report(args.output)
    
    print(f"\nAnalyzed {len(analyzer.results)} commodities")
    print(f"Report saved to: {report_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
