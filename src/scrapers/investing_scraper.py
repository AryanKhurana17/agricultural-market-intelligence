"""
Investing.com / Yahoo Finance Commodities Scraper
=================================================

Scrapes real agricultural commodity price data using yfinance (Yahoo Finance)
as a reliable source for market data.

Usage:
    from src.scrapers.investing_scraper import InvestingScraper
    scraper = InvestingScraper()
    data = scraper.scrape_all_commodities()
"""

import time
import random
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.utils.logger import get_logger
from src.utils import config

logger = get_logger(__name__)

# Yahoo Finance Tickers for Commodities
COMMODITY_TICKERS = {
    'WHEAT': 'ZW=F',      # Chicago SRW Wheat Futures
    'CORN': 'ZC=F',       # Corn Futures
    'RICE': 'ZR=F',       # Rough Rice Futures
    'SOYBEANS': 'ZS=F',   # Soybean Futures
    'COFFEE': 'KC=F',     # Coffee Futures
    'SUGAR': 'SB=F',      # Sugar #11 Futures
    'COTTON': 'CT=F'      # Cotton #2 Futures
}

class InvestingScraper:
    """
    Scraper for Commodity prices using yfinance.
    """
    
    def __init__(self):
        """Initialize the scraper."""
        self.data: Dict[str, List[Dict]] = {}
        self.rate_limit_min = 2.0
        self.rate_limit_max = 5.0
    
    def _rate_limit(self):
        """Apply rate limiting delay."""
        delay = random.uniform(self.rate_limit_min, self.rate_limit_max)
        time.sleep(delay)
    
    def scrape_commodity(
        self,
        commodity: str,
        period: str = "5y"
    ) -> List[Dict]:
        """
        Scrape real data for a single commodity from Yahoo Finance.
        
        Args:
            commodity: Commodity name (e.g., 'WHEAT')
            period: Data period (1y, 2y, 5y, 10y, max)
            
        Returns:
            List of price records
        """
        ticker_symbol = COMMODITY_TICKERS.get(commodity.upper())
        if not ticker_symbol:
            logger.warning(f"No ticker found for {commodity}")
            return []
            
        logger.info(f"Fetching real data for {commodity} ({ticker_symbol})...")
        
        try:
            # Fetch data from yfinance
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data returned for {commodity}")
                return []
            
            records = []
            # Reset index to get Date as a column
            hist = hist.reset_index()
            
            for _, row in hist.iterrows():
                # Handle timezone if present
                date_val = row['Date']
                if hasattr(date_val, 'date'):
                    date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val).split(' ')[0]
                
                records.append({
                    'commodity': commodity.title(),
                    'date': date_str,
                    'price': round(float(row['Close']), 2),
                    'open': round(float(row['Open']), 2),
                    'high': round(float(row['High']), 2),
                    'low': round(float(row['Low']), 2),
                    'volume': int(row['Volume']),
                    'unit': 'USD',  # Futures are generally in USD
                    'source': 'Yahoo Finance',
                    'market': 'Global Futures'
                })
            
            logger.info(f"Successfully fetched {len(records)} records for {commodity}")
            return records
            
        except Exception as e:
            logger.error(f"Error fetching data for {commodity}: {e}")
            return []
    
    def scrape_all_commodities(self, period: str = "10y") -> Dict[str, List[Dict]]:
        """
        Scrape data for all configured commodities.
        
        Args:
            period: History period
            
        Returns:
            Dictionary mapping commodity names to their data
        """
        logger.info(f"Starting generic scraper for {len(COMMODITY_TICKERS)} commodities")
        
        results = {}
        for commodity in COMMODITY_TICKERS:
            data = self.scrape_commodity(commodity, period)
            if data:
                self.data[commodity] = data
                results[commodity] = data
            self._rate_limit()
            
        return results

    def save_to_json(self, output_dir: str) -> str:
        """
        Save scraped data to JSON file.
        
        Args:
            output_dir: Directory to save file
            
        Returns:
            Path to saved file
        """
        import json
        import os
        
        if not self.data:
            logger.warning("No data to save")
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"investing_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
            
        logger.info(f"Saved data to {filepath}")
        return filepath

if __name__ == "__main__":
    # Test run
    scraper = InvestingScraper()
    data = scraper.scrape_all_commodities(period="1mo")
    print(f"Scraped {len(data)} commodities")
