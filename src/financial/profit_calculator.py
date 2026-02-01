"""
Profit Calculator for Agricultural Production
==============================================

This module calculates profitability metrics for agricultural commodities
using current market prices and production costs.

Calculations:
- Revenue = market_price × yield
- Profit = revenue - total_cost
- Profit margin = (profit / revenue) × 100
- ROI = (profit / total_cost) × 100
- Profit per day = profit / cycle_days
- Annual profit = profit × cycles_per_year

Author: INNOFarms Data Analyst Assignment
Date: 2026-02-01
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"profit_calculator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# DATABASE ACCESS
# =============================================================================
class DatabaseAccess:
    """Database access layer for fetching current prices."""
    
    def __init__(self, db_path: str = "database/market_data.db"):
        """Initialize database access."""
        self.db_path = db_path
    
    def get_current_price(self, commodity: str) -> Optional[float]:
        """
        Get the most recent average price for a commodity.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Current price per quintal or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT AVG(price) as avg_price
                FROM price_data p
                JOIN commodities c ON p.commodity_id = c.commodity_id
                WHERE c.name = ?
                    AND p.date = (
                        SELECT MAX(date) FROM price_data p2
                        JOIN commodities c2 ON p2.commodity_id = c2.commodity_id
                        WHERE c2.name = ?
                    )
            """, (commodity, commodity))
            row = cursor.fetchone()
            return row['avg_price'] if row and row['avg_price'] else None
        finally:
            conn.close()
    
    def get_avg_price(self, commodity: str, days: int = 30) -> Optional[float]:
        """
        Get average price over a period.
        
        Args:
            commodity: Commodity name
            days: Number of days to average
            
        Returns:
            Average price or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("""
                SELECT AVG(price) as avg_price
                FROM price_data p
                JOIN commodities c ON p.commodity_id = c.commodity_id
                WHERE c.name = ?
                    AND p.date >= date('now', ?)
            """, (commodity, f'-{days} days'))
            row = cursor.fetchone()
            return row['avg_price'] if row and row['avg_price'] else None
        finally:
            conn.close()


# =============================================================================
# PROFIT CALCULATOR
# =============================================================================
class ProfitCalculator:
    """
    Calculates profitability metrics for agricultural commodities.
    """
    
    def __init__(
        self, 
        config_path: str = "config/production_costs.yaml",
        db_path: str = "database/market_data.db"
    ):
        """
        Initialize the profit calculator.
        
        Args:
            config_path: Path to production costs configuration
            db_path: Path to market data database
        """
        self.logger = setup_logging()
        self.config = self._load_config(config_path)
        self.commodities = self.config.get('commodities', {})
        self.db = DatabaseAccess(db_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load production costs configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            return {}
    
    def calculate_profitability(
        self, 
        commodity: str, 
        market_price: Optional[float] = None
    ) -> Dict:
        """
        Calculate profitability metrics for a commodity.
        
        Args:
            commodity: Commodity name
            market_price: Override market price (optional)
            
        Returns:
            Dictionary with profitability metrics
        """
        commodity_key = commodity.lower()
        
        if commodity_key not in self.commodities:
            return {'commodity': commodity, 'error': 'Unknown commodity'}
        
        commodity_data = self.commodities[commodity_key]
        costs = commodity_data.get('costs', {})
        
        # Get production parameters
        total_cost = costs.get('total', 0)
        yield_qty = commodity_data.get('yield_per_hectare', 0)
        cycle_days = commodity_data.get('cycle_days', 1)
        
        if yield_qty == 0:
            return {'commodity': commodity, 'error': 'Invalid yield data'}
        
        # Get market price
        if market_price is None:
            # Try to get from database
            db_price = self.db.get_avg_price(commodity_data.get('name', commodity), 30)
            if db_price:
                market_price = db_price
            else:
                # Use default from config
                market_price = self.config.get('market_prices', {}).get(commodity_key, 0)
        
        if market_price == 0:
            return {'commodity': commodity, 'error': 'No price data available'}
        
        # Calculate metrics per hectare
        revenue = market_price * yield_qty
        profit = revenue - total_cost
        
        # Avoid division by zero
        profit_margin = (profit / revenue * 100) if revenue > 0 else 0
        roi = (profit / total_cost * 100) if total_cost > 0 else 0
        
        # Efficiency metrics
        profit_per_day = profit / cycle_days
        cycles_per_year = 365 / cycle_days
        annual_profit = profit * cycles_per_year
        
        # Break-even analysis
        breakeven_price = total_cost / yield_qty if yield_qty > 0 else 0
        breakeven_yield = total_cost / market_price if market_price > 0 else 0
        margin_of_safety = ((market_price - breakeven_price) / market_price * 100) if market_price > 0 else 0
        
        return {
            'commodity': commodity_data.get('name', commodity),
            'category': commodity_data.get('category', 'Unknown'),
            'cycle_days': cycle_days,
            
            # Input metrics
            'market_price': round(market_price, 2),
            'yield_per_hectare': yield_qty,
            'total_cost': total_cost,
            
            # Profitability metrics (per hectare)
            'revenue': round(revenue, 2),
            'profit': round(profit, 2),
            'profit_margin': round(profit_margin, 2),
            'roi': round(roi, 2),
            
            # Efficiency metrics
            'profit_per_day': round(profit_per_day, 2),
            'cycles_per_year': round(cycles_per_year, 2),
            'annual_profit': round(annual_profit, 2),
            
            # Break-even analysis
            'breakeven_price': round(breakeven_price, 2),
            'breakeven_yield': round(breakeven_yield, 2),
            'margin_of_safety': round(margin_of_safety, 2),
            
            # Cost per unit
            'cost_per_quintal': round(total_cost / yield_qty, 2),
            
            # Profitability flag
            'is_profitable': profit > 0,
            
            'unit': {
                'price': 'INR/quintal',
                'cost': 'INR/hectare',
                'profit': 'INR/hectare'
            }
        }
    
    def calculate_all_profitability(self) -> Dict[str, Dict]:
        """
        Calculate profitability for all commodities.
        
        Returns:
            Dictionary mapping commodity names to their profitability
        """
        results = {}
        
        for commodity_key in self.commodities:
            commodity_name = self.commodities[commodity_key].get('name', commodity_key)
            results[commodity_name] = self.calculate_profitability(commodity_key)
        
        return results
    
    def rank_by_profitability(self, metric: str = 'roi') -> List[Dict]:
        """
        Rank commodities by a profitability metric.
        
        Args:
            metric: Metric to rank by (roi, profit_margin, profit_per_day, annual_profit)
            
        Returns:
            List of commodities sorted by the metric (descending)
        """
        all_profit = self.calculate_all_profitability()
        
        ranked = []
        for commodity, data in all_profit.items():
            if 'error' not in data:
                ranked.append({
                    'commodity': commodity,
                    'metric_value': data.get(metric, 0),
                    'profit': data.get('profit', 0),
                    'roi': data.get('roi', 0),
                    'profit_margin': data.get('profit_margin', 0),
                    'annual_profit': data.get('annual_profit', 0)
                })
        
        ranked.sort(key=lambda x: x['metric_value'], reverse=True)
        
        return ranked
    
    def generate_report(self, output_path: str = "reports/profitability_report.json") -> str:
        """
        Generate comprehensive profitability report.
        
        Returns:
            Path to generated report
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        all_profit = self.calculate_all_profitability()
        rankings = self.rank_by_profitability('roi')
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'profitability': all_profit,
            'rankings': {
                'by_roi': rankings,
                'top_commodity': rankings[0] if rankings else None
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to: {output_path}")
        return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agricultural Profit Calculator')
    parser.add_argument('--commodity', type=str, default=None,
                        help='Specific commodity to analyze')
    parser.add_argument('--price', type=float, default=None,
                        help='Override market price')
    
    args = parser.parse_args()
    
    print("📊 Agricultural Profit Calculator")
    print("=" * 50)
    
    calculator = ProfitCalculator()
    
    if args.commodity:
        result = calculator.calculate_profitability(args.commodity, args.price)
        print(f"\n{args.commodity} Profitability Analysis:")
        print(json.dumps(result, indent=2))
    else:
        rankings = calculator.rank_by_profitability('roi')
        print("\nProfitability Rankings (by ROI):")
        print("-" * 60)
        for i, item in enumerate(rankings, 1):
            status = "✅" if item['profit'] > 0 else "❌"
            print(f"{i}. {item['commodity']:15} | ROI: {item['roi']:>7.1f}% | "
                  f"Profit: ₹{item['profit']:>10,.0f}/ha {status}")
        print("-" * 60)


if __name__ == "__main__":
    main()
