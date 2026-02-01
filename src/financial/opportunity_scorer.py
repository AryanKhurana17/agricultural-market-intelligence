"""
Opportunity Scorer for Agricultural Commodities
=================================================

This module scores and ranks agricultural commodities based on
investment opportunity using a weighted multi-criteria approach.

Scoring Criteria:
- Profit margin (30%)
- ROI (25%)
- Profit per day (20%)
- Price stability (15%)
- Demand trend (10%)

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

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.financial.profit_calculator import ProfitCalculator
from src.analytics.trend_analyzer import TrendAnalyzer

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"opportunity_scorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
# OPPORTUNITY SCORER
# =============================================================================
class OpportunityScorer:
    """
    Scores and ranks agricultural commodities as investment opportunities.
    """
    
    # Default scoring weights
    DEFAULT_WEIGHTS = {
        'profit_margin': 0.30,
        'roi': 0.25,
        'profit_per_day': 0.20,
        'price_stability': 0.15,
        'demand_trend': 0.10
    }
    
    def __init__(
        self, 
        config_path: str = "config/production_costs.yaml",
        db_path: str = "database/market_data.db"
    ):
        """
        Initialize the opportunity scorer.
        
        Args:
            config_path: Path to production costs configuration
            db_path: Path to market data database
        """
        self.logger = setup_logging()
        self.config = self._load_config(config_path)
        self.weights = self.config.get('scoring_weights', self.DEFAULT_WEIGHTS)
        
        # Initialize calculators
        self.profit_calc = ProfitCalculator(config_path, db_path)
        self.trend_analyzer = TrendAnalyzer(db_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-100 scale."""
        if max_val == min_val:
            return 50.0
        return ((value - min_val) / (max_val - min_val)) * 100
    
    def score_commodity(self, commodity: str) -> Dict:
        """
        Score a single commodity.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Dictionary with scores and final ranking score
        """
        # Get profitability metrics
        profit_data = self.profit_calc.calculate_profitability(commodity)
        
        if 'error' in profit_data:
            return {'commodity': commodity, 'error': profit_data['error']}
        
        # Get trend analysis
        trend_data = self.trend_analyzer.analyze_commodity(commodity)
        
        # Calculate individual scores (0-100)
        scores = {}
        
        # Profit margin score (higher is better)
        profit_margin = profit_data.get('profit_margin', 0)
        scores['profit_margin'] = min(100, max(0, profit_margin * 2))  # 50% margin = 100 score
        
        # ROI score (higher is better)
        roi = profit_data.get('roi', 0)
        scores['roi'] = min(100, max(0, roi * 0.5))  # 200% ROI = 100 score
        
        # Profit per day score (normalized later)
        scores['profit_per_day_raw'] = profit_data.get('profit_per_day', 0)
        
        # Price stability score (lower volatility is better)
        volatility = trend_data.get('volatility_pct', 20) if 'error' not in trend_data else 20
        scores['price_stability'] = 100 - min(100, volatility * 3)  # 33% volatility = 0 score
        
        # Demand trend score (upward trend is better)
        trend = trend_data.get('trend', 'Stable') if 'error' not in trend_data else 'Stable'
        trend_strength = abs(trend_data.get('trend_strength', 0)) if 'error' not in trend_data else 0
        
        if trend == 'Upward':
            scores['demand_trend'] = min(100, 50 + trend_strength * 5)
        elif trend == 'Downward':
            scores['demand_trend'] = max(0, 50 - trend_strength * 5)
        else:
            scores['demand_trend'] = 50  # Neutral
        
        return {
            'commodity': profit_data.get('commodity', commodity),
            'category': profit_data.get('category', 'Unknown'),
            'scores': scores,
            'metrics': {
                'profit_margin': profit_data.get('profit_margin', 0),
                'roi': profit_data.get('roi', 0),
                'profit_per_day': profit_data.get('profit_per_day', 0),
                'annual_profit': profit_data.get('annual_profit', 0),
                'volatility_pct': volatility,
                'trend': trend,
                'trend_strength': trend_strength
            },
            'profitability': profit_data
        }
    
    def rank_opportunities(self) -> List[Dict]:
        """
        Score and rank all commodities.
        
        Returns:
            List of commodities sorted by opportunity score
        """
        commodities = list(self.config.get('commodities', {}).keys())
        all_scores = []
        
        for commodity in commodities:
            score_data = self.score_commodity(commodity)
            if 'error' not in score_data:
                all_scores.append(score_data)
        
        if not all_scores:
            return []
        
        # Normalize profit_per_day scores
        ppd_values = [s['scores']['profit_per_day_raw'] for s in all_scores]
        ppd_min, ppd_max = min(ppd_values), max(ppd_values)
        
        for score_data in all_scores:
            score_data['scores']['profit_per_day'] = self._normalize_score(
                score_data['scores']['profit_per_day_raw'],
                ppd_min, ppd_max
            )
        
        # Calculate weighted final score
        for score_data in all_scores:
            final_score = 0
            for metric, weight in self.weights.items():
                if metric in score_data['scores']:
                    final_score += score_data['scores'][metric] * weight
            
            score_data['opportunity_score'] = round(final_score, 2)
            
            # Generate recommendation
            if final_score >= 70:
                score_data['recommendation'] = 'Strong Buy'
                score_data['risk_level'] = 'Low'
            elif final_score >= 50:
                score_data['recommendation'] = 'Buy'
                score_data['risk_level'] = 'Medium'
            elif final_score >= 30:
                score_data['recommendation'] = 'Hold'
                score_data['risk_level'] = 'Medium-High'
            else:
                score_data['recommendation'] = 'Avoid'
                score_data['risk_level'] = 'High'
        
        # Sort by opportunity score
        all_scores.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        # Add rank
        for i, score_data in enumerate(all_scores):
            score_data['rank'] = i + 1
        
        return all_scores
    
    def generate_report(self, output_path: str = "reports/opportunity_rankings.json") -> str:
        """
        Generate opportunity ranking report.
        
        Returns:
            Path to generated report
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        rankings = self.rank_opportunities()
        
        # Summary
        if rankings:
            top_picks = [r for r in rankings if r['recommendation'] in ['Strong Buy', 'Buy']]
            avoid = [r for r in rankings if r['recommendation'] == 'Avoid']
        else:
            top_picks = []
            avoid = []
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_commodities': len(rankings),
            'scoring_weights': self.weights,
            'rankings': rankings,
            'summary': {
                'top_picks': [r['commodity'] for r in top_picks],
                'avoid': [r['commodity'] for r in avoid],
                'best_opportunity': rankings[0] if rankings else None
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
    
    parser = argparse.ArgumentParser(description='Agricultural Opportunity Scorer')
    parser.add_argument('--output', type=str, default='reports/opportunity_rankings.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    print("🎯 Agricultural Opportunity Scorer")
    print("=" * 60)
    
    scorer = OpportunityScorer()
    rankings = scorer.rank_opportunities()
    
    print("\nOpportunity Rankings:")
    print("-" * 60)
    for r in rankings:
        print(f"{r['rank']}. {r['commodity']:15} | Score: {r['opportunity_score']:>5.1f} | "
              f"{r['recommendation']:12} | Risk: {r['risk_level']}")
    print("-" * 60)
    
    # Save report
    output_path = scorer.generate_report(args.output)
    print(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    main()
