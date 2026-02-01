"""
USDA NASS QuickStats API Scraper
================================

Scrapes agricultural commodity data from USDA QuickStats API.
Supports multiple commodities with configurable date ranges.

Usage:
    from src.scrapers.usda_scraper import USDANASSScraper
    scraper = USDANASSScraper()
    data = scraper.scrape_all_commodities()
"""

import os
import sys
import json
import time
import random
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================
USDA_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

# Commodity configurations - easily extendable
COMMODITIES = {
    "WHEAT": {
        "commodity_desc": "WHEAT",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / BU"
    },
    "CORN": {
        "commodity_desc": "CORN",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / BU"
    },
    "RICE": {
        "commodity_desc": "RICE",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / CWT"
    },
    "SOYBEANS": {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / BU"
    },
    "COTTON": {
        "commodity_desc": "COTTON",
        "statisticcat_desc": "PRICE RECEIVED",
        "unit_desc": "$ / LB"
    }
}


# =============================================================================
# USDA SCRAPER CLASS
# =============================================================================
class USDANASSScraper:
    """
    Scraper for USDA NASS QuickStats API.
    
    Attributes:
        api_key: USDA API key
        base_url: API endpoint URL
        data: Scraped data storage
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the USDA scraper.
        
        Args:
            api_key: USDA API key (defaults to env variable)
        """
        self.api_key = api_key or config.USDA_API_KEY
        self.base_url = USDA_API_URL
        self.data: Dict[str, List[Dict]] = {}
        self.rate_limit_min = config.RATE_LIMIT_MIN
        self.rate_limit_max = config.RATE_LIMIT_MAX
        
        if not self.api_key:
            logger.warning("USDA API key not set")
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """
        Make API request with error handling.
        
        Args:
            params: Query parameters
            
        Returns:
            JSON response or None on error
        """
        params['key'] = self.api_key
        params['format'] = 'JSON'
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def _rate_limit(self):
        """Apply rate limiting delay."""
        delay = random.uniform(self.rate_limit_min, self.rate_limit_max)
        time.sleep(delay)
    
    def scrape_commodity(
        self,
        commodity: str,
        years: int = 10,
        commodity_config: Dict = None
    ) -> List[Dict]:
        """
        Scrape data for a single commodity.
        
        Args:
            commodity: Commodity name
            years: Years of historical data
            commodity_config: Custom commodity configuration
            
        Returns:
            List of price records
        """
        if commodity_config is None:
            commodity_config = COMMODITIES.get(commodity.upper(), {})
        
        if not commodity_config:
            logger.warning(f"No configuration for commodity: {commodity}")
            return []
        
        current_year = datetime.now().year
        start_year = current_year - years
        
        params = {
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "commodity_desc": commodity_config.get("commodity_desc", commodity),
            "statisticcat_desc": commodity_config.get("statisticcat_desc", "PRICE RECEIVED"),
            "year__GE": str(start_year),
            "year__LE": str(current_year),
            "freq_desc": "ANNUAL"
        }
        
        logger.info(f"Fetching {commodity} data ({start_year}-{current_year})...")
        
        response = self._make_request(params)
        
        if not response or 'data' not in response:
            logger.warning(f"No data returned for {commodity}")
            return []
        
        records = []
        for item in response['data']:
            try:
                value = item.get('Value', '').replace(',', '')
                if value and value != '(D)' and value != '(NA)':
                    record = {
                        'commodity': commodity.title(),
                        'date': f"{item.get('year', datetime.now().year)}-01-01",
                        'price': float(value),
                        'unit': item.get('unit_desc', 'USD'),
                        'source': 'USDA NASS',
                        'state': item.get('state_name', 'US TOTAL'),
                        'market': 'National Average',
                        'frequency': item.get('freq_desc', 'ANNUAL')
                    }
                    records.append(record)
            except (ValueError, KeyError) as e:
                logger.debug(f"Skipping invalid record: {e}")
                continue
        
        logger.info(f"Scraped {len(records)} records for {commodity}")
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
        
        logger.info(f"Starting USDA scraper for {len(commodities)} commodities")
        
        total_records = 0
        for commodity in commodities:
            records = self.scrape_commodity(commodity.upper(), years)
            if records:
                self.data[commodity.title()] = records
                total_records += len(records)
        
        logger.info(f"USDA scraping complete: {total_records} total records")
        
        return {
            'source': 'USDA NASS',
            'commodities_scraped': len(self.data),
            'total_records': total_records,
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
        filename = Path(output_dir) / f"usda_{timestamp}.json"
        
        output = {
            'source': 'USDA NASS',
            'scraped_at': datetime.now().isoformat(),
            'api_url': self.base_url,
            'commodities': self.data
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved data to {filename}")
        return str(filename)
    
    def add_commodity(self, name: str, config_dict: Dict) -> None:
        """
        Add a new commodity configuration for scraping.
        
        Args:
            name: Commodity name
            config_dict: Configuration dictionary
        """
        COMMODITIES[name.upper()] = config_dict
        logger.info(f"Added commodity: {name}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='USDA NASS Scraper')
    parser.add_argument('--api-key', type=str, default=None,
                        help='USDA API key')
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
    print("USDA NASS Data Scraper")
    print("=" * 50)
    
    scraper = USDANASSScraper(api_key=args.api_key)
    result = scraper.scrape_all_commodities(commodities, args.years)
    
    output_file = scraper.save_to_json(args.output)
    
    print(f"\nResults:")
    print(f"  Commodities: {result['commodities_scraped']}")
    print(f"  Records: {result['total_records']}")
    print(f"  Output: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
