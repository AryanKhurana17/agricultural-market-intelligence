"""
Main Analysis Runner
====================

Runs all analysis modules and generates reports.

Usage:
    python src/main_analysis.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.config import config

# Import analysis modules
sys.path.insert(0, str(PROJECT_ROOT / "src" / "analytics"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "financial"))

from trend_analyzer import TrendAnalyzer
from forecasters import ForecastEngine
from cost_calculator import CostCalculator
from profit_calculator import ProfitCalculator
from opportunity_scorer import OpportunityScorer
from risk_analyzer import RiskAnalyzer
from scenario_planner import ScenarioPlanner

logger = get_logger(__name__)


# =============================================================================
# ANALYSIS RUNNER
# =============================================================================
class AnalysisRunner:
    """Runs all analysis modules."""
    
    def __init__(
        self,
        config_path: str = "config/production_costs.yaml",
        db_path: str = None,
        reports_dir: str = "reports"
    ):
        self.config_path = config_path
        self.db_path = db_path or config.DATABASE_PATH
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.trend_analyzer = TrendAnalyzer(self.db_path)
        self.forecaster = ForecastEngine(self.db_path)
        self.cost_calc = CostCalculator(config_path)
        self.profit_calc = ProfitCalculator(config_path, self.db_path)
        self.opportunity_scorer = OpportunityScorer(config_path, self.db_path)
        self.risk_analyzer = RiskAnalyzer(config_path, self.db_path)
        self.scenario_planner = ScenarioPlanner(config_path, self.db_path)
    
    def run_all(self) -> Dict[str, Any]:
        """Run all analysis modules."""
        logger.info("=" * 60)
        logger.info("Starting Analysis Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        results = {}
        
        # 1. Trend Analysis
        print("\n[1/7] Running Trend Analysis...")
        try:
            results['trend'] = self.trend_analyzer.generate_report(
                str(self.reports_dir / "trend_analysis.json")
            )
            print("      OK")
        except Exception as e:
            print(f"      Error: {e}")
            logger.error(f"Trend analysis failed: {e}")
        
        # 2. Forecasting
        print("[2/7] Running Price Forecasting...")
        try:
            results['forecast'] = self.forecaster.generate_report(
                str(self.reports_dir / "forecast_output.json")
            )
            print("      OK")
        except Exception as e:
            print(f"      Error: {e}")
            logger.error(f"Forecasting failed: {e}")
        
        # 3. Cost Analysis
        print("[3/7] Running Cost Analysis...")
        try:
            cost_data = self.cost_calc.get_cost_summary()
            cost_file = str(self.reports_dir / "cost_analysis.json")
            with open(cost_file, 'w') as f:
                json.dump(cost_data, f, indent=2)
            results['cost'] = cost_file
            print("      OK")
        except Exception as e:
            print(f"      Error: {e}")
            logger.error(f"Cost analysis failed: {e}")
        
        # 4. Profitability
        print("[4/7] Running Profitability Analysis...")
        try:
            results['profit'] = self.profit_calc.generate_report(
                str(self.reports_dir / "profitability_report.json")
            )
            print("      OK")
        except Exception as e:
            print(f"      Error: {e}")
            logger.error(f"Profitability analysis failed: {e}")
        
        # 5. Opportunity Scoring
        print("[5/7] Running Opportunity Scoring...")
        try:
            results['opportunity'] = self.opportunity_scorer.generate_report(
                str(self.reports_dir / "opportunity_rankings.json")
            )
            print("      OK")
        except Exception as e:
            print(f"      Error: {e}")
            logger.error(f"Opportunity scoring failed: {e}")
        
        # 6. Risk Analysis
        print("[6/7] Running Risk Analysis...")
        try:
            results['risk'] = self.risk_analyzer.generate_report(
                str(self.reports_dir / "risk_analysis.json")
            )
            print("      OK")
        except Exception as e:
            print(f"      Error: {e}")
            logger.error(f"Risk analysis failed: {e}")
        
        # 7. Scenario Planning
        print("[7/7] Running Scenario Planning...")
        try:
            results['scenario'] = self.scenario_planner.generate_report(
                str(self.reports_dir / "scenario_analysis.json")
            )
            print("      OK")
        except Exception as e:
            print(f"      Error: {e}")
            logger.error(f"Scenario planning failed: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Summary
        summary = {
            'generated_at': datetime.now().isoformat(),
            'duration_seconds': round(duration, 2),
            'reports': results
        }
        
        summary_file = str(self.reports_dir / "analysis_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"Analysis Complete ({duration:.2f}s)")
        logger.info("=" * 60)
        
        return summary


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='INNOFarms Analysis Runner')
    parser.add_argument('--config', type=str, default='config/production_costs.yaml')
    parser.add_argument('--db', type=str, default=None)
    parser.add_argument('--reports', type=str, default='reports')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INNOFarms Agricultural Market Intelligence")
    print("=" * 60)
    
    runner = AnalysisRunner(
        config_path=args.config,
        db_path=args.db,
        reports_dir=args.reports
    )
    
    summary = runner.run_all()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Duration: {summary.get('duration_seconds', 0):.2f}s")
    print(f"Reports: {args.reports}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
