"""
Risk Analyzer for Agricultural Investments
============================================

This module calculates risk metrics and provides risk-adjusted
recommendations for agricultural commodities.

Risk Metrics:
- Volatility Risk (price fluctuation)
- Weather Risk (seasonal/climate impact)
- Market Risk (price correlation)
- Overall Risk Score

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
import numpy as np

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"risk_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    """Database access layer."""
    
    def __init__(self, db_path: str = "database/market_data.db"):
        self.db_path = db_path
    
    def get_price_series(self, commodity: str, days: int = 365) -> List[float]:
        """Get price series for volatility calculation."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT AVG(price) as price
                FROM price_data p
                JOIN commodities c ON p.commodity_id = c.commodity_id
                WHERE c.name = ?
                    AND p.date >= date('now', ?)
                GROUP BY p.date
                ORDER BY p.date
            """, (commodity, f'-{days} days'))
            prices = [row[0] for row in cursor.fetchall() if row[0]]
            return prices
        finally:
            conn.close()


# =============================================================================
# RISK ANALYZER
# =============================================================================
class RiskAnalyzer:
    """
    Analyzes and quantifies risk for agricultural investments.
    """
    
    # Base risk factors from config
    DEFAULT_RISK_FACTORS = {
        'weather_risk': 0.15,
        'pest_risk': 0.10,
        'market_risk': 0.20,
        'storage_loss': 0.05
    }
    
    def __init__(
        self, 
        config_path: str = "config/production_costs.yaml",
        db_path: str = "database/market_data.db"
    ):
        """Initialize the risk analyzer."""
        self.logger = setup_logging()
        self.config = self._load_config(config_path)
        self.db = DatabaseAccess(db_path)
        self.risk_factors = self.config.get('risk_factors', self.DEFAULT_RISK_FACTORS)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def calculate_volatility_risk(self, commodity: str, days: int = 365) -> Dict:
        """
        Calculate price volatility risk.
        
        Args:
            commodity: Commodity name
            days: Days of history
            
        Returns:
            Dictionary with volatility metrics
        """
        prices = self.db.get_price_series(commodity, days)
        
        if len(prices) < 10:
            return {
                'volatility_pct': 0,
                'risk_level': 'Unknown',
                'risk_score': 50
            }
        
        prices = np.array(prices)
        
        # Calculate coefficient of variation
        mean_price = np.mean(prices)
        std_dev = np.std(prices)
        cv = (std_dev / mean_price * 100) if mean_price > 0 else 0
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Risk level classification
        if cv < 10:
            risk_level = 'Low'
            risk_score = 25
        elif cv < 20:
            risk_level = 'Medium'
            risk_score = 50
        elif cv < 30:
            risk_level = 'High'
            risk_score = 75
        else:
            risk_level = 'Very High'
            risk_score = 100
        
        return {
            'volatility_pct': round(cv, 2),
            'std_dev': round(std_dev, 2),
            'max_drawdown': round(max_drawdown, 2),
            'risk_level': risk_level,
            'risk_score': risk_score
        }
    
    def calculate_commodity_risk(self, commodity: str) -> Dict:
        """
        Calculate comprehensive risk profile for a commodity.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Dictionary with all risk metrics
        """
        commodity_key = commodity.lower()
        commodity_config = self.config.get('commodities', {}).get(commodity_key, {})
        
        # Volatility risk from historical data
        vol_risk = self.calculate_volatility_risk(commodity)
        
        # Category-based risks (vegetables are higher risk)
        category = commodity_config.get('category', 'Unknown')
        if category == 'Vegetables':
            category_risk = 70
            weather_multiplier = 1.3
        else:
            category_risk = 40
            weather_multiplier = 1.0
        
        # Calculate component risks
        weather_risk = self.risk_factors['weather_risk'] * 100 * weather_multiplier
        pest_risk = self.risk_factors['pest_risk'] * 100
        market_risk = min(100, vol_risk['risk_score'] * self.risk_factors['market_risk'] / 0.20)
        storage_risk = self.risk_factors['storage_loss'] * 100
        
        if category == 'Vegetables':
            storage_risk *= 2  # Perishable goods
        
        # Overall risk score (weighted average)
        overall_risk = (
            vol_risk['risk_score'] * 0.35 +
            weather_risk * 0.25 +
            market_risk * 0.20 +
            category_risk * 0.10 +
            storage_risk * 0.10
        )
        
        # Risk classification
        if overall_risk < 30:
            overall_level = 'Low'
        elif overall_risk < 50:
            overall_level = 'Medium'
        elif overall_risk < 70:
            overall_level = 'High'
        else:
            overall_level = 'Very High'
        
        return {
            'commodity': commodity,
            'category': category,
            'risk_components': {
                'volatility_risk': round(vol_risk['risk_score'], 2),
                'weather_risk': round(weather_risk, 2),
                'market_risk': round(market_risk, 2),
                'category_risk': round(category_risk, 2),
                'storage_risk': round(storage_risk, 2)
            },
            'volatility_details': vol_risk,
            'overall_risk_score': round(overall_risk, 2),
            'overall_risk_level': overall_level,
            'mitigation_suggestions': self._get_mitigation_suggestions(vol_risk['risk_level'], category)
        }
    
    def _get_mitigation_suggestions(self, vol_level: str, category: str) -> List[str]:
        """Generate risk mitigation suggestions."""
        suggestions = []
        
        if vol_level in ['High', 'Very High']:
            suggestions.append("Consider price hedging through forward contracts")
            suggestions.append("Diversify across multiple commodities")
        
        if category == 'Vegetables':
            suggestions.append("Plan for cold storage facilities")
            suggestions.append("Establish direct market linkages to reduce time to sale")
        
        suggestions.append("Monitor weather forecasts for planting decisions")
        suggestions.append("Consider crop insurance to mitigate losses")
        
        return suggestions
    
    def analyze_all_commodities(self) -> Dict[str, Dict]:
        """Analyze risk for all commodities."""
        commodities = list(self.config.get('commodities', {}).keys())
        results = {}
        
        for commodity in commodities:
            commodity_name = self.config['commodities'][commodity].get('name', commodity)
            results[commodity_name] = self.calculate_commodity_risk(commodity_name)
        
        return results
    
    def calculate_portfolio_risk(self, allocation: Dict[str, float]) -> Dict:
        """
        Calculate portfolio risk given allocation.
        
        Args:
            allocation: Dict mapping commodity to allocation percentage
            
        Returns:
            Portfolio risk metrics
        """
        total_allocation = sum(allocation.values())
        if total_allocation == 0:
            return {'error': 'Empty allocation'}
        
        # Normalize allocation
        allocation = {k: v / total_allocation for k, v in allocation.items()}
        
        weighted_risk = 0
        commodity_risks = {}
        
        for commodity, weight in allocation.items():
            risk_data = self.calculate_commodity_risk(commodity)
            if 'error' not in risk_data:
                commodity_risks[commodity] = {
                    'weight': round(weight * 100, 2),
                    'risk_score': risk_data['overall_risk_score'],
                    'weighted_contribution': round(risk_data['overall_risk_score'] * weight, 2)
                }
                weighted_risk += risk_data['overall_risk_score'] * weight
        
        # Diversification benefit (simplified)
        num_commodities = len([w for w in allocation.values() if w > 0.05])
        diversification_factor = 1 - (0.05 * min(num_commodities, 5))
        adjusted_risk = weighted_risk * diversification_factor
        
        return {
            'portfolio_risk_score': round(adjusted_risk, 2),
            'undiversified_risk': round(weighted_risk, 2),
            'diversification_benefit': round((weighted_risk - adjusted_risk), 2),
            'commodity_contributions': commodity_risks,
            'risk_level': 'Low' if adjusted_risk < 35 else 
                         'Medium' if adjusted_risk < 55 else
                         'High' if adjusted_risk < 75 else 'Very High'
        }
    
    def generate_report(self, output_path: str = "reports/risk_analysis.json") -> str:
        """Generate risk analysis report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        all_risks = self.analyze_all_commodities()
        
        # Sort by risk score
        sorted_risks = sorted(
            all_risks.items(), 
            key=lambda x: x[1].get('overall_risk_score', 100)
        )
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'risk_factors': self.risk_factors,
            'commodity_risks': all_risks,
            'rankings': {
                'lowest_risk': [name for name, _ in sorted_risks[:2]],
                'highest_risk': [name for name, _ in sorted_risks[-2:]]
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
    
    parser = argparse.ArgumentParser(description='Agricultural Risk Analyzer')
    parser.add_argument('--commodity', type=str, default=None,
                        help='Specific commodity to analyze')
    
    args = parser.parse_args()
    
    print("⚠️  Agricultural Risk Analyzer")
    print("=" * 60)
    
    analyzer = RiskAnalyzer()
    
    if args.commodity:
        result = analyzer.calculate_commodity_risk(args.commodity)
        print(f"\n{args.commodity} Risk Analysis:")
        print(json.dumps(result, indent=2))
    else:
        all_risks = analyzer.analyze_all_commodities()
        print("\nRisk Summary:")
        print("-" * 60)
        for name, data in sorted(all_risks.items(), key=lambda x: x[1]['overall_risk_score']):
            print(f"{name:15} | Risk Score: {data['overall_risk_score']:>5.1f} | "
                  f"Level: {data['overall_risk_level']}")
        print("-" * 60)
        
        # Save report
        output_path = analyzer.generate_report()
        print(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    main()
