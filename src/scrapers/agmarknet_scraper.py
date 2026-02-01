"""
AGMARKNET Scraper
=================

Scrapes agricultural price data from AGMARKNET (agmarknet.gov.in).
Generates sample data as AGMARKNET requires browser automation.

Usage:
    from src.scrapers.agmarknet_scraper import AGMARKNETScraper
    scraper = AGMARKNETScraper()
    data = scraper.scrape_all_commodities()
"""

import os
import sys
import json
import time
import random
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
        "group": "Cereals",
        "name": "Wheat",
        "base_price": 2400,
        "unit": "Rs./Quintal"
    },
    "RICE": {
        "group": "Cereals",
        "name": "Rice",
        "base_price": 2800,
        "unit": "Rs./Quintal"
    },
    "MAIZE": {
        "group": "Cereals",
        "name": "Maize",
        "base_price": 1800,
        "unit": "Rs./Quintal"
    },
    "TOMATO": {
        "group": "Vegetables",
        "name": "Tomato",
        "base_price": 2500,
        "unit": "Rs./Quintal"
    },
    "ONION": {
        "group": "Vegetables",
        "name": "Onion",
        "base_price": 1800,
        "unit": "Rs./Quintal"
    }
}

# Indian states and markets for sample data
MARKETS = {
    "Andhra Pradesh": ["Tirupati APMC", "Vijayawada APMC"],
    "Gujarat": ["Ahmedabad APMC", "Rajkot APMC"],
    "Madhya Pradesh": ["Indore APMC", "Bhopal APMC"],
    "Maharashtra": ["Mumbai APMC", "Pune APMC"],
    "Punjab": ["Amritsar APMC", "Ludhiana APMC"],
    "Uttar Pradesh": ["Lucknow APMC", "Kanpur APMC"]
}


# =============================================================================
# AGMARKNET SCRAPER CLASS
# =============================================================================
class AGMARKNETScraper:
    """
    Scraper for AGMARKNET Indian agricultural market prices.
    
    Note: Uses sample data as AGMARKNET requires Selenium for live scraping.
    """
    
    def __init__(self, use_sample_data: bool = True):
        """
        Initialize the AGMARKNET scraper.
        
        Args:
            use_sample_data: Whether to use generated sample data
        """
        self.use_sample_data = use_sample_data
        self.data: Dict[str, List[Dict]] = {}
        self.rate_limit_min = config.RATE_LIMIT_MIN
        self.rate_limit_max = config.RATE_LIMIT_MAX
    
    def _rate_limit(self):
        """Apply rate limiting delay."""
        delay = random.uniform(self.rate_limit_min, self.rate_limit_max)
        time.sleep(delay)
    
    def _generate_sample_data(
        self,
        commodity: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Generate realistic sample APMC price data.
        
        Args:
            commodity: Commodity name
            days: Days of data to generate
            
        Returns:
            List of price records
        """
        commodity_config = COMMODITIES.get(commodity.upper(), {})
        base_price = commodity_config.get('base_price', 2000)
        
        records = []
        end_date = datetime.now()
        
        for day_offset in range(days):
            date = end_date - timedelta(days=day_offset)
            
            for state, markets in MARKETS.items():
                for market in markets:
                    # Add variation by market and day
                    market_factor = random.uniform(0.95, 1.05)
                    day_factor = random.uniform(0.97, 1.03)
                    
                    modal_price = base_price * market_factor * day_factor
                    min_price = modal_price * random.uniform(0.85, 0.95)
                    max_price = modal_price * random.uniform(1.05, 1.15)
                    
                    # Random arrivals
                    arrivals = random.uniform(50, 500)
                    
                    records.append({
                        'commodity': commodity.title(),
                        'date': date.strftime('%Y-%m-%d'),
                        'source': 'AGMARKNET',
                        'state': state,
                        'market': market,
                        'arrivals': round(arrivals, 2),
                        'arrivals_unit': 'Tonnes',
                        'variety': random.choice(['Local', 'Grade A', 'Premium']),
                        'grade': random.choice(['A', 'B', 'FAQ']),
                        'min_price': round(min_price, 2),
                        'max_price': round(max_price, 2),
                        'modal_price': round(modal_price, 2),
                        'unit': commodity_config.get('unit', 'Rs./Quintal')
                    })
        
        return records
    
    def scrape_commodity(
        self,
        commodity: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Scrape data for a single commodity.
        
        Args:
            commodity: Commodity name
            days: Days of historical data
            
        Returns:
            List of price records
        """
        commodity_upper = commodity.upper()
        
        if commodity_upper not in COMMODITIES:
            logger.warning(f"Unknown commodity: {commodity}")
            return []
        
        logger.info(f"Fetching {commodity} data ({days} days)...")
        
        if self.use_sample_data:
            records = self._generate_sample_data(commodity, days)
        else:
            logger.warning("Live scraping not implemented, using sample data")
            records = self._generate_sample_data(commodity, days)
        
        logger.info(f"Scraped {len(records)} records for {commodity}")
        self._rate_limit()
        
        return records
    
    def scrape_all_commodities(
        self,
        commodities: List[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Scrape data for multiple commodities.
        
        Args:
            commodities: List of commodity names (None = all)
            days: Days of historical data
            
        Returns:
            Summary of scraping results
        """
        if commodities is None:
            commodities = list(COMMODITIES.keys())
        
        logger.info(f"Starting AGMARKNET scraper for {len(commodities)} commodities")
        
        total_records = 0
        for commodity in commodities:
            records = self.scrape_commodity(commodity, days)
            if records:
                self.data[commodity.title()] = records
                total_records += len(records)
        
        logger.info(f"AGMARKNET scraping complete: {total_records} total records")
        
        return {
            'source': 'AGMARKNET',
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
        filename = Path(output_dir) / f"agmarknet_{timestamp}.json"
        
        output = {
            'source': 'AGMARKNET',
            'scraped_at': datetime.now().isoformat(),
            'url': 'https://agmarknet.gov.in',
            'is_sample_data': self.use_sample_data,
            'commodities': self.data
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved data to {filename}")
        return str(filename)
    
    def add_commodity(
        self,
        name: str,
        group: str,
        base_price: float,
        unit: str = "Rs./Quintal"
    ) -> None:
        """
        Add a new commodity configuration.
        
        Args:
            name: Commodity name
            group: Category (Cereals, Vegetables, etc.)
            base_price: Base price for sample generation
            unit: Price unit
        """
        COMMODITIES[name.upper()] = {
            'group': group,
            'name': name.title(),
            'base_price': base_price,
            'unit': unit
        }
        logger.info(f"Added commodity: {name}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AGMARKNET Scraper')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of historical data')
    parser.add_argument('--commodities', type=str, default=None,
                        help='Commodities to scrape (comma-separated)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    commodities = None
    if args.commodities:
        commodities = [c.strip() for c in args.commodities.split(',')]
    
    print("=" * 50)
    print("AGMARKNET Data Scraper")
    print("=" * 50)
    
    scraper = AGMARKNETScraper()
    result = scraper.scrape_all_commodities(commodities, args.days)
    
    output_file = scraper.save_to_json(args.output)
    
    print(f"\nResults:")
    print(f"  Commodities: {result['commodities_scraped']}")
    print(f"  Records: {result['total_records']}")
    print(f"  Sample Data: {result['sample_data']}")
    print(f"  Output: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
