"""
Main Scraper Runner
===================

Runs all data scrapers in sequence and saves raw data.

Usage:
    python src/scrapers/main_scraper.py [--sources all|usda|investing|agmarknet]
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# =============================================================================
# SCRAPER IMPORTS
# =============================================================================
def import_scrapers():
    """Import scrapers dynamically to handle import errors gracefully."""
    scrapers = {}
    
    try:
        from src.scrapers.usda_scraper import USDANASSScraper
        scrapers['usda'] = USDANASSScraper
        logger.info("USDA scraper loaded")
    except ImportError as e:
        logger.warning(f"USDA scraper not available: {e}")
    
    try:
        from src.scrapers.investing_scraper import InvestingScraper
        scrapers['investing'] = InvestingScraper
        logger.info("Investing.com scraper loaded")
    except ImportError as e:
        logger.warning(f"Investing scraper not available: {e}")
    
    try:
        from src.scrapers.agmarknet_scraper import AGMARKNETScraper
        scrapers['agmarknet'] = AGMARKNETScraper
        logger.info("AGMARKNET scraper loaded")
    except ImportError as e:
        logger.warning(f"AGMARKNET scraper not available: {e}")
    
    return scrapers


# =============================================================================
# MAIN SCRAPER ORCHESTRATOR
# =============================================================================
class ScraperRunner:
    """
    Orchestrates running multiple data scrapers.
    
    Attributes:
        output_dir: Directory for raw data output
        available_scrapers: Dict of available scraper classes
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the scraper runner.
        
        Args:
            output_dir: Output directory for raw data
        """
        self.output_dir = Path(output_dir or config.DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.available_scrapers = import_scrapers()
        self.results: Dict[str, Any] = {}
    
    def run_usda(self, commodities: List[str] = None, years: int = 10) -> Dict:
        """
        Run USDA NASS API scraper.
        
        Args:
            commodities: List of commodities to scrape
            years: Number of years of historical data
            
        Returns:
            Scraping results
        """
        if 'usda' not in self.available_scrapers:
            logger.error("USDA scraper not available")
            return {'status': 'error', 'message': 'Scraper not available'}
        
        if not config.USDA_API_KEY:
            logger.error("USDA_API_KEY not set in environment")
            return {'status': 'error', 'message': 'API key not configured'}
        
        logger.info("Starting USDA NASS scraper...")
        
        try:
            scraper = self.available_scrapers['usda'](api_key=config.USDA_API_KEY)
            result = scraper.scrape_all_commodities(commodities, years)
            
            # Save to file
            output_file = scraper.save_to_json(str(self.output_dir))
            
            self.results['usda'] = {
                'status': 'success',
                'records': result.get('total_records', 0),
                'output_file': output_file
            }
            logger.info(f"USDA scraper completed: {result.get('total_records', 0)} records")
            
        except Exception as e:
            logger.error(f"USDA scraper failed: {e}")
            self.results['usda'] = {'status': 'error', 'message': str(e)}
        
        return self.results.get('usda', {})
    
    def run_investing(self, commodities: List[str] = None, years: int = 10) -> Dict:
        """
        Run Investing.com scraper.
        
        Args:
            commodities: List of commodities to scrape
            years: Number of years of historical data
            
        Returns:
            Scraping results
        """
        if 'investing' not in self.available_scrapers:
            logger.error("Investing scraper not available")
            return {'status': 'error', 'message': 'Scraper not available'}
        
        logger.info("Starting Investing.com scraper...")
        
        try:
            scraper = self.available_scrapers['investing']()
            # Investing scraper now uses yfinance and fixed tickers, supports 'period'
            result = scraper.scrape_all_commodities(period=f"{years}y" if years <= 10 else "max")
            
            output_file = scraper.save_to_json(str(self.output_dir))
            
            self.results['investing'] = {
                'status': 'success',
                'records': len(scraper.data) * 100, # Approximate or need better count from result
                'output_file': output_file
            }
            # Calculate total records if result is a dict of lists
            total_records = sum(len(records) for records in result.values())
            
            self.results['investing']['records'] = total_records
            logger.info(f"Investing scraper completed: {total_records} records")
            
        except Exception as e:
            logger.error(f"Investing scraper failed: {e}")
            self.results['investing'] = {'status': 'error', 'message': str(e)}
        
        return self.results.get('investing', {})
    
    def run_agmarknet(self, commodities: List[str] = None, days: int = 7) -> Dict:
        """
        Run AGMARKNET scraper.
        
        Args:
            commodities: List of commodities to scrape
            days: Number of days of historical data
            
        Returns:
            Scraping results
        """
        if 'agmarknet' not in self.available_scrapers:
            logger.error("AGMARKNET scraper not available")
            return {'status': 'error', 'message': 'Scraper not available'}
        
        logger.info("Starting AGMARKNET scraper...")
        
        try:
            scraper = self.available_scrapers['agmarknet']()
            # Real Agmarknet scraper defines its own commodity list internally for reliability
            result = scraper.scrape_all_commodities()
            
            output_file = scraper.save_to_json(str(self.output_dir))
            
            total_records = sum(len(records) for records in result.values())
            
            self.results['agmarknet'] = {
                'status': 'success',
                'records': total_records,
                'output_file': output_file
            }
            logger.info(f"AGMARKNET scraper completed: {total_records} records")
            
        except Exception as e:
            logger.error(f"AGMARKNET scraper failed: {e}")
            self.results['agmarknet'] = {'status': 'error', 'message': str(e)}
        
        return self.results.get('agmarknet', {})
    
    def run_all(
        self,
        commodities: List[str] = None,
        years: int = 10,
        sources: List[str] = None
    ) -> Dict:
        """
        Run all available scrapers.
        
        Args:
            commodities: List of commodities to scrape
            years: Number of years of historical data
            sources: Specific sources to run (default: all)
            
        Returns:
            Combined results from all scrapers
        """
        logger.info("=" * 60)
        logger.info("Starting All Scrapers")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        if sources is None:
            sources = ['usda', 'investing', 'agmarknet']
        
        # Run each requested scraper
        if 'usda' in sources:
            self.run_usda(commodities, years)
        
        if 'investing' in sources:
            self.run_investing(commodities, years)
        
        if 'agmarknet' in sources:
            self.run_agmarknet(commodities, days=7)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Summary
        total_records = sum(
            r.get('records', 0) for r in self.results.values()
            if r.get('status') == 'success'
        )
        
        summary = {
            'duration_seconds': round(duration, 2),
            'total_records': total_records,
            'sources': self.results
        }
        
        logger.info("=" * 60)
        logger.info(f"Scraping Complete: {total_records} records in {duration:.2f}s")
        logger.info("=" * 60)
        
        return summary
    
    def get_summary(self) -> Dict:
        """Get summary of all scraping results."""
        return self.results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for the scraper runner."""
    parser = argparse.ArgumentParser(
        description='INNOFarms Data Scraper Runner'
    )
    parser.add_argument(
        '--sources',
        type=str,
        default='all',
        help='Sources to scrape: all, usda, investing, agmarknet (comma-separated)'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Number of years of historical data'
    )
    parser.add_argument(
        '--commodities',
        type=str,
        default=None,
        help='Commodities to scrape (comma-separated)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for raw data'
    )
    
    args = parser.parse_args()
    
    # Parse sources
    if args.sources.lower() == 'all':
        sources = None  # Will run all
    else:
        sources = [s.strip().lower() for s in args.sources.split(',')]
    
    # Parse commodities
    commodities = None
    if args.commodities:
        commodities = [c.strip() for c in args.commodities.split(',')]
    
    # Run scrapers
    print("=" * 60)
    print("INNOFarms Data Scraper")
    print("=" * 60)
    
    runner = ScraperRunner(output_dir=args.output)
    summary = runner.run_all(
        commodities=commodities,
        years=args.years,
        sources=sources
    )
    
    # Print summary
    print("\nScraping Summary:")
    print("-" * 40)
    for source, result in summary.get('sources', {}).items():
        status = result.get('status', 'unknown')
        if status == 'success':
            print(f"  {source}: {result.get('records', 0)} records")
        else:
            print(f"  {source}: {status} - {result.get('message', '')}")
    print("-" * 40)
    print(f"Total: {summary.get('total_records', 0)} records")
    print(f"Duration: {summary.get('duration_seconds', 0):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
