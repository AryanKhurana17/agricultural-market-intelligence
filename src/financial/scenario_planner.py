"""
Scenario Planner for Agricultural Investments
===============================================

This module performs scenario analysis for agricultural investments,
exploring different market conditions and their impact on profitability.

Scenarios:
- Optimistic: 20% price increase
- Base Case: Current prices
- Pessimistic: 20% price decrease
- Stress Test: 40% price decrease

Author: INNOFarms Data Analyst Assignment
Date: 2026-02-01
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.financial.profit_calculator import ProfitCalculator

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"scenario_planner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
# SCENARIO DEFINITIONS
# =============================================================================
SCENARIOS = {
    'optimistic': {
        'name': 'Optimistic',
        'description': 'Bull market with 20% price increase',
        'price_multiplier': 1.20,
        'probability': 0.20
    },
    'base': {
        'name': 'Base Case',
        'description': 'Current market conditions',
        'price_multiplier': 1.00,
        'probability': 0.50
    },
    'pessimistic': {
        'name': 'Pessimistic',
        'description': 'Bear market with 20% price decrease',
        'price_multiplier': 0.80,
        'probability': 0.25
    },
    'stress': {
        'name': 'Stress Test',
        'description': 'Severe downturn with 40% price decrease',
        'price_multiplier': 0.60,
        'probability': 0.05
    }
}


# =============================================================================
# SCENARIO PLANNER
# =============================================================================
class ScenarioPlanner:
    """
    Performs scenario analysis for agricultural investments.
    """
    
    def __init__(
        self, 
        config_path: str = "config/production_costs.yaml",
        db_path: str = "database/market_data.db"
    ):
        """Initialize the scenario planner."""
        self.logger = setup_logging()
        self.config = self._load_config(config_path)
        self.profit_calc = ProfitCalculator(config_path, db_path)
        self.scenarios = SCENARIOS
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def analyze_scenario(
        self, 
        commodity: str, 
        scenario_key: str,
        base_price: Optional[float] = None
    ) -> Dict:
        """
        Analyze a commodity under a specific scenario.
        
        Args:
            commodity: Commodity name
            scenario_key: Scenario key from SCENARIOS
            base_price: Override base price (optional)
            
        Returns:
            Profitability under the scenario
        """
        scenario = self.scenarios.get(scenario_key, self.scenarios['base'])
        
        # Get base case profitability
        base_result = self.profit_calc.calculate_profitability(commodity, base_price)
        
        if 'error' in base_result:
            return {'commodity': commodity, 'scenario': scenario['name'], 'error': base_result['error']}
        
        # Calculate scenario price
        scenario_price = base_result['market_price'] * scenario['price_multiplier']
        
        # Recalculate with scenario price
        scenario_result = self.profit_calc.calculate_profitability(commodity, scenario_price)
        
        return {
            'commodity': scenario_result['commodity'],
            'scenario': scenario['name'],
            'description': scenario['description'],
            'probability': scenario['probability'],
            
            # Price comparison
            'base_price': base_result['market_price'],
            'scenario_price': round(scenario_price, 2),
            'price_change': f"{(scenario['price_multiplier'] - 1) * 100:+.0f}%",
            
            # Profitability
            'revenue': scenario_result['revenue'],
            'profit': scenario_result['profit'],
            'roi': scenario_result['roi'],
            'profit_margin': scenario_result['profit_margin'],
            'is_profitable': scenario_result['is_profitable'],
            
            # Comparison to base
            'profit_vs_base': round(scenario_result['profit'] - base_result['profit'], 2),
            'roi_vs_base': round(scenario_result['roi'] - base_result['roi'], 2)
        }
    
    def analyze_all_scenarios(self, commodity: str) -> Dict[str, Dict]:
        """
        Analyze a commodity under all scenarios.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Dictionary mapping scenario keys to results
        """
        results = {}
        
        for scenario_key in self.scenarios:
            results[scenario_key] = self.analyze_scenario(commodity, scenario_key)
        
        # Calculate expected value (probability-weighted)
        expected_profit = sum(
            results[k].get('profit', 0) * self.scenarios[k]['probability']
            for k in self.scenarios
        )
        expected_roi = sum(
            results[k].get('roi', 0) * self.scenarios[k]['probability']
            for k in self.scenarios
        )
        
        results['expected_value'] = {
            'expected_profit': round(expected_profit, 2),
            'expected_roi': round(expected_roi, 2)
        }
        
        return results
    
    def generate_scenario_matrix(self) -> Dict[str, Dict[str, Dict]]:
        """
        Generate scenario analysis for all commodities.
        
        Returns:
            Nested dictionary: commodity -> scenario -> results
        """
        commodities = list(self.config.get('commodities', {}).keys())
        matrix = {}
        
        for commodity in commodities:
            commodity_name = self.config['commodities'][commodity].get('name', commodity)
            matrix[commodity_name] = self.analyze_all_scenarios(commodity_name)
        
        return matrix
    
    def calculate_breakeven_scenarios(self, commodity: str) -> Dict:
        """
        Calculate at what price level the commodity breaks even.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Break-even analysis
        """
        base_result = self.profit_calc.calculate_profitability(commodity)
        
        if 'error' in base_result:
            return {'error': base_result['error']}
        
        current_price = base_result['market_price']
        breakeven_price = base_result['breakeven_price']
        
        # Calculate how much price can fall before breaking even
        max_decline = ((current_price - breakeven_price) / current_price) * 100 if current_price > 0 else 0
        
        # Stress test at different price points
        stress_points = [0.90, 0.80, 0.70, 0.60, 0.50]
        stress_results = []
        
        for multiplier in stress_points:
            test_price = current_price * multiplier
            profit_at_price = self.profit_calc.calculate_profitability(commodity, test_price)
            stress_results.append({
                'price_level': f"{multiplier * 100:.0f}%",
                'price': round(test_price, 2),
                'profit': profit_at_price.get('profit', 0),
                'roi': profit_at_price.get('roi', 0),
                'is_profitable': profit_at_price.get('is_profitable', False)
            })
        
        return {
            'commodity': base_result['commodity'],
            'current_price': current_price,
            'breakeven_price': breakeven_price,
            'margin_of_safety': round(max_decline, 2),
            'max_decline_before_loss': f"{max_decline:.1f}%",
            'stress_test': stress_results
        }
    
    def generate_portfolio_recommendation(
        self, 
        investment_amount: float = 100000,
        risk_tolerance: str = 'medium'
    ) -> Dict:
        """
        Generate portfolio recommendation based on scenario analysis.
        
        Args:
            investment_amount: Total investment amount
            risk_tolerance: 'low', 'medium', or 'high'
            
        Returns:
            Portfolio recommendation
        """
        matrix = self.generate_scenario_matrix()
        
        # Score commodities
        recommendations = []
        
        for commodity, scenarios in matrix.items():
            if 'expected_value' not in scenarios:
                continue
            
            expected_roi = scenarios['expected_value']['expected_roi']
            expected_profit = scenarios['expected_value']['expected_profit']
            
            # Get stress test survival
            stress = scenarios.get('stress', {})
            survives_stress = stress.get('is_profitable', False)
            
            # Calculate score based on risk tolerance
            if risk_tolerance == 'low':
                # Prioritize stability
                score = expected_roi * 0.3 + (100 if survives_stress else 0) * 0.7
            elif risk_tolerance == 'high':
                # Prioritize returns
                optimistic = scenarios.get('optimistic', {})
                score = expected_roi * 0.5 + optimistic.get('roi', 0) * 0.5
            else:  # medium
                score = expected_roi * 0.6 + (50 if survives_stress else 0) * 0.4
            
            recommendations.append({
                'commodity': commodity,
                'expected_roi': expected_roi,
                'expected_profit': expected_profit,
                'survives_stress': survives_stress,
                'score': round(score, 2)
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Allocate investment
        top_picks = recommendations[:3]  # Top 3 commodities
        total_score = sum(r['score'] for r in top_picks)
        
        if total_score > 0:
            allocation = {
                r['commodity']: {
                    'percentage': round((r['score'] / total_score) * 100, 1),
                    'amount': round((r['score'] / total_score) * investment_amount, 2),
                    'expected_return': round(r['expected_profit'] * (r['score'] / total_score), 2)
                }
                for r in top_picks
            }
        else:
            allocation = {}
        
        return {
            'investment_amount': investment_amount,
            'risk_tolerance': risk_tolerance,
            'recommendations': recommendations,
            'portfolio_allocation': allocation,
            'expected_portfolio_return': sum(
                a['expected_return'] for a in allocation.values()
            )
        }
    
    def generate_report(self, output_path: str = "reports/scenario_analysis.json") -> str:
        """Generate comprehensive scenario analysis report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        matrix = self.generate_scenario_matrix()
        
        # Breakeven analysis
        commodities = list(self.config.get('commodities', {}).keys())
        breakeven = {}
        for commodity in commodities:
            commodity_name = self.config['commodities'][commodity].get('name', commodity)
            breakeven[commodity_name] = self.calculate_breakeven_scenarios(commodity_name)
        
        # Portfolio recommendations
        portfolios = {
            'conservative': self.generate_portfolio_recommendation(100000, 'low'),
            'balanced': self.generate_portfolio_recommendation(100000, 'medium'),
            'aggressive': self.generate_portfolio_recommendation(100000, 'high')
        }
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'scenarios_defined': list(self.scenarios.keys()),
            'scenario_matrix': matrix,
            'breakeven_analysis': breakeven,
            'portfolio_recommendations': portfolios
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
    
    parser = argparse.ArgumentParser(description='Agricultural Scenario Planner')
    parser.add_argument('--commodity', type=str, default=None,
                        help='Specific commodity to analyze')
    parser.add_argument('--investment', type=float, default=100000,
                        help='Investment amount for portfolio')
    
    args = parser.parse_args()
    
    print("📊 Agricultural Scenario Planner")
    print("=" * 60)
    
    planner = ScenarioPlanner()
    
    if args.commodity:
        scenarios = planner.analyze_all_scenarios(args.commodity)
        print(f"\n{args.commodity} Scenario Analysis:")
        for key, data in scenarios.items():
            if key != 'expected_value':
                print(f"\n{data['scenario']} ({data['price_change']}):")
                print(f"  Profit: ₹{data['profit']:,.0f}/ha  |  ROI: {data['roi']:.1f}%")
        
        ev = scenarios['expected_value']
        print(f"\nExpected Value: ₹{ev['expected_profit']:,.0f}/ha  |  ROI: {ev['expected_roi']:.1f}%")
    else:
        # Generate full report
        output_path = planner.generate_report()
        
        # Show portfolio recommendations
        portfolio = planner.generate_portfolio_recommendation(args.investment)
        print("\n📈 Portfolio Recommendation (Balanced):")
        print("-" * 60)
        for commodity, alloc in portfolio['portfolio_allocation'].items():
            print(f"{commodity:15} | {alloc['percentage']:>5.1f}% | ₹{alloc['amount']:>10,.0f}")
        print("-" * 60)
        print(f"Expected Return: ₹{portfolio['expected_portfolio_return']:,.0f}")
        print(f"\n✅ Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
