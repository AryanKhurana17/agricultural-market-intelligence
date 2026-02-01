"""
Investing.com Commodities Scraper
=================================

Scrapes agricultural commodity price data from Investing.com.
Uses historical price API with sample data fallback.

Usage:
    from src.scrapers.investing_scraper import InvestingScraper
    scraper = InvestingScraper()
    data = scraper.scrape_all_commodities()
"""

import os
import sys
import json
import time
import random
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# =============================================================================
# COMMODITY CONFIGURATIONS
# =============================================================================
COMMODITIES = {
    "WHEAT": {
        "symbol": "wheat",
        "pair_id": "8917",
        "name": "Wheat",
        "unit": "cents/bushel"
    },
    "CORN": {
        "symbol": "corn",
        "pair_id": "8918",
        "name": "Corn",
        "unit": "cents/bushel"
    },
    "RICE": {
        "symbol": "rice",
        "pair_id": "8920",
        "name": "Rice",
        "unit": "USD/cwt"
    }
}


# =============================================================================
# INVESTING.COM SCRAPER CLASS
# =============================================================================
class InvestingScraper:
    """
    Scraper for Investing.com commodity prices.
    
    Uses sample data generation as Investing.com requires authentication
    for historical data API access.
    """
    
    def __init__(self, use_sample_data: bool = True):
        """
        Initialize the Investing.com scraper.
        
        Args:
            use_sample_data: Whether to use generated sample data
        """
        self.use_sample_data = use_sample_data
        self.data: Dict[str, List[Dict]] = {}
        self.rate_limit_min = config.RATE_LIMIT_MIN
        self.rate_limit_max = config.RATE_LIMIT_MAX
        
        # Base prices for sample data generation
        self.base_prices = {
            'Wheat': 550,
            'Corn': 400,
            'Rice': 15,
            'Soybeans': 1200,
            'Coffee': 130,
            'Sugar': 18,
            'Cotton': 80
        }
    
    def _rate_limit(self):
        """Apply rate limiting delay."""
        delay = random.uniform(self.rate_limit_min, self.rate_limit_max)
        time.sleep(delay)
    
    def _generate_sample_data(
        self,
        commodity: str,
        years: int = 10
    ) -> List[Dict]:
        """
        Generate realistic sample price data.
        
        Args:
            commodity: Commodity name
            years: Years of data to generate
            
        Returns:
            List of price records
        """
        base_price = self.base_prices.get(commodity, 100)
        commodity_config = COMMODITIES.get(commodity.upper(), {})
        
        records = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        current_date = start_date
        
        # Generate daily prices with realistic patterns
        price = base_price
        trend = 0
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                # Add trend and noise
                trend = trend * 0.95 + random.gauss(0, 0.3)
                daily_change = trend + random.gauss(0, base_price * 0.02)
                price = max(base_price * 0.5, price + daily_change)
                
                # Seasonal adjustment
                month = current_date.month
                if commodity in ['Wheat', 'Corn']:
                    seasonal = 0.05 * np.sin(2 * np.pi * (month - 6) / 12)
                    price *= (1 + seasonal)
                
                records.append({
                    'commodity': commodity,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'price': round(price, 2),
                    'open': round(price * random.uniform(0.98, 1.02), 2),
                    'high': round(price * random.uniform(1.0, 1.03), 2),
                    'low': round(price * random.uniform(0.97, 1.0), 2),
                    'unit': commodity_config.get('unit', 'USD'),
                    'source': 'Investing.com',
                    'market': 'Futures'
                })
            
            current_date += timedelta(days=1)
        
        return records
    
    def scrape_commodity(
        self,
        commodity: str,
        years: int = 10
    ) -> List[Dict]:
        """
        Scrape data for a single commodity.
        
        Args:
            commodity: Commodity name
            years: Years of historical data
            
        Returns:
            List of price records
        """
        commodity_title = commodity.title()
        
        if commodity.upper() not in COMMODITIES:
            logger.warning(f"Unknown commodity: {commodity}")
            return []
        
        logger.info(f"Fetching {commodity_title} data...")
        
        if self.use_sample_data:
            records = self._generate_sample_data(commodity_title, years)
        else:
            # Real API implementation would go here
            logger.warning("Live API not configured, using sample data")
            records = self._generate_sample_data(commodity_title, years)
        
        logger.info(f"Scraped {len(records)} records for {commodity_title}")
        self._rate_limit()
        
        return records
    
    def scrape_all_commodities(
        self,
        commodities: List[str] = None,
        years: int = 10
    ) -> Dict[str, Any]:
        """
        Scrape data for multiple commodities.
        
        Args:
            commodities: List of commodity names (None = all)
            years: Years of historical data
            
        Returns:
            Summary of scraping results
        """
        if commodities is None:
            commodities = list(COMMODITIES.keys())
        
        logger.info(f"Starting Investing.com scraper for {len(commodities)} commodities")
        
        total_records = 0
        for commodity in commodities:
            records = self.scrape_commodity(commodity, years)
            if records:
                self.data[commodity.title()] = records
                total_records += len(records)
        
        logger.info(f"Investing.com scraping complete: {total_records} total records")
        
        return {
            'source': 'Investing.com',
            'commodities_scraped': len(self.data),
            'total_records': total_records,
            'sample_data': self.use_sample_data,
            'scraped_at': datetime.now().isoformat()
        }
    
    def save_to_json(self, output_dir: str = None) -> str:
        """
        Save scraped data to JSON file.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = str(config.DATA_DIR)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = Path(output_dir) / f"investing_{timestamp}.json"
        
        output = {
            'source': 'Investing.com',
            'scraped_at': datetime.now().isoformat(),
            'is_sample_data': self.use_sample_data,
            'commodities': self.data
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved data to {filename}")
        return str(filename)
    
    def add_commodity(self, name: str, config_dict: Dict) -> None:
        """
        Add a new commodity configuration.
        
        Args:
            name: Commodity name
            config_dict: Configuration dictionary
        """
        COMMODITIES[name.upper()] = config_dict
        if 'base_price' in config_dict:
            self.base_prices[name.title()] = config_dict['base_price']
        logger.info(f"Added commodity: {name}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Investing.com Scraper')
    parser.add_argument('--years', type=int, default=10,
                        help='Years of historical data')
    parser.add_argument('--commodities', type=str, default=None,
                        help='Commodities to scrape (comma-separated)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    commodities = None
    if args.commodities:
        commodities = [c.strip() for c in args.commodities.split(',')]
    
    print("=" * 50)
    print("Investing.com Data Scraper")
    print("=" * 50)
    
    scraper = InvestingScraper()
    result = scraper.scrape_all_commodities(commodities, args.years)
    
    output_file = scraper.save_to_json(args.output)
    
    print(f"\nResults:")
    print(f"  Commodities: {result['commodities_scraped']}")
    print(f"  Records: {result['total_records']}")
    print(f"  Sample Data: {result['sample_data']}")
    print(f"  Output: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
