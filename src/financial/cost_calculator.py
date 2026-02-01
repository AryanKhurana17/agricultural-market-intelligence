"""
Cost Calculator for Agricultural Production
============================================

This module calculates production costs per hectare for agricultural
commodities based on the production_costs.yaml configuration.

Features:
- Per hectare cost breakdown
- Cost comparison across commodities
- Cost sensitivity analysis

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

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"cost_calculator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
# COST CALCULATOR
# =============================================================================
class CostCalculator:
    """
    Calculates production costs for agricultural commodities.
    """
    
    def __init__(self, config_path: str = "config/production_costs.yaml"):
        """
        Initialize the cost calculator.
        
        Args:
            config_path: Path to production costs configuration
        """
        self.logger = setup_logging()
        self.config = self._load_config(config_path)
        self.commodities = self.config.get('commodities', {})
    
    def _load_config(self, config_path: str) -> Dict:
        """Load production costs configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            return {}
    
    def get_commodity_costs(self, commodity: str) -> Dict:
        """
        Get cost breakdown for a specific commodity.
        
        Args:
            commodity: Commodity name (lowercase)
            
        Returns:
            Dictionary with cost breakdown
        """
        commodity_key = commodity.lower()
        
        if commodity_key not in self.commodities:
            self.logger.warning(f"Unknown commodity: {commodity}")
            return {}
        
        commodity_data = self.commodities[commodity_key]
        costs = commodity_data.get('costs', {})
        
        return {
            'commodity': commodity_data.get('name', commodity),
            'category': commodity_data.get('category', 'Unknown'),
            'cycle_days': commodity_data.get('cycle_days', 0),
            'yield_per_hectare': commodity_data.get('yield_per_hectare', 0),
            'seasons': commodity_data.get('seasons', []),
            'cost_breakdown': {
                'seeds': costs.get('seeds', 0),
                'fertilizers': costs.get('fertilizers', 0),
                'pesticides': costs.get('pesticides', 0),
                'irrigation': costs.get('irrigation', 0),
                'labor': costs.get('labor', 0),
                'machinery': costs.get('machinery', 0),
                'transport': costs.get('transport', 0),
                'miscellaneous': costs.get('miscellaneous', 0),
            },
            'total_cost': costs.get('total', 0),
            'unit': 'INR/hectare'
        }
    
    def get_cost_per_quintal(self, commodity: str) -> float:
        """
        Calculate production cost per quintal.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Cost per quintal in INR
        """
        costs = self.get_commodity_costs(commodity)
        
        if not costs or costs.get('yield_per_hectare', 0) == 0:
            return 0.0
        
        total_cost = costs['total_cost']
        yield_qty = costs['yield_per_hectare']
        
        return round(total_cost / yield_qty, 2)
    
    def compare_costs(self) -> List[Dict]:
        """
        Compare costs across all commodities.
        
        Returns:
            List of commodity cost comparisons sorted by efficiency
        """
        comparisons = []
        
        for commodity_key in self.commodities:
            costs = self.get_commodity_costs(commodity_key)
            cost_per_quintal = self.get_cost_per_quintal(commodity_key)
            
            if costs:
                comparisons.append({
                    'commodity': costs['commodity'],
                    'total_cost_per_hectare': costs['total_cost'],
                    'yield_per_hectare': costs['yield_per_hectare'],
                    'cost_per_quintal': cost_per_quintal,
                    'cycle_days': costs['cycle_days'],
                    'cost_per_day': round(costs['total_cost'] / costs['cycle_days'], 2)
                })
        
        # Sort by cost per quintal (most efficient first)
        comparisons.sort(key=lambda x: x['cost_per_quintal'])
        
        return comparisons
    
    def get_cost_summary(self) -> Dict:
        """
        Generate a summary of all production costs.
        
        Returns:
            Dictionary with cost summary
        """
        comparisons = self.compare_costs()
        
        if not comparisons:
            return {}
        
        # Find extremes
        lowest_cost = comparisons[0]
        highest_cost = comparisons[-1]
        
        # Calculate averages
        avg_cost_hectare = sum(c['total_cost_per_hectare'] for c in comparisons) / len(comparisons)
        avg_cost_quintal = sum(c['cost_per_quintal'] for c in comparisons) / len(comparisons)
        
        return {
            'total_commodities': len(comparisons),
            'comparisons': comparisons,
            'most_efficient': {
                'commodity': lowest_cost['commodity'],
                'cost_per_quintal': lowest_cost['cost_per_quintal']
            },
            'least_efficient': {
                'commodity': highest_cost['commodity'],
                'cost_per_quintal': highest_cost['cost_per_quintal']
            },
            'averages': {
                'avg_cost_per_hectare': round(avg_cost_hectare, 2),
                'avg_cost_per_quintal': round(avg_cost_quintal, 2)
            }
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agricultural Production Cost Calculator')
    parser.add_argument('--commodity', type=str, default=None,
                        help='Specific commodity to analyze')
    
    args = parser.parse_args()
    
    print("💰 Agricultural Production Cost Calculator")
    print("=" * 50)
    
    calculator = CostCalculator()
    
    if args.commodity:
        costs = calculator.get_commodity_costs(args.commodity)
        cost_per_quintal = calculator.get_cost_per_quintal(args.commodity)
        print(f"\n{args.commodity} Cost Analysis:")
        print(json.dumps(costs, indent=2))
        print(f"\nCost per Quintal: ₹{cost_per_quintal}")
    else:
        summary = calculator.get_cost_summary()
        print("\nCost Comparison (sorted by efficiency):")
        print("-" * 60)
        for comp in summary.get('comparisons', []):
            print(f"{comp['commodity']:15} | ₹{comp['cost_per_quintal']:>8}/quintal | "
                  f"₹{comp['total_cost_per_hectare']:>8}/ha | {comp['cycle_days']} days")
        print("-" * 60)
        print(f"\n✅ Most efficient: {summary['most_efficient']['commodity']} "
              f"(₹{summary['most_efficient']['cost_per_quintal']}/quintal)")


if __name__ == "__main__":
    main()
