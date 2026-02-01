"""
ETL Pipeline for Agricultural Market Data
=========================================

Extract, Transform, Load pipeline for processing scraped data
and loading it into SQLite database.

Usage:
    python src/etl/etl_pipeline.py [--input data/raw] [--db database/market_data.db]
"""

import os
import sys
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================
PRICE_UNIT_CONVERSIONS = {
    '$ / BU': {'factor': 1.0, 'target': 'USD/Bushel'},
    '$ / CWT': {'factor': 1.0, 'target': 'USD/CWT'},
    'cents/bushel': {'factor': 0.01, 'target': 'USD/Bushel'},
    'cents/lb': {'factor': 0.01, 'target': 'USD/lb'},
    'USD/cwt': {'factor': 1.0, 'target': 'USD/CWT'},
    'Rs./Quintal': {'factor': 1.0, 'target': 'INR/Quintal'},
    'Rs./Kg': {'factor': 100.0, 'target': 'INR/Quintal'}
}

SOURCE_MAPPINGS = {
    'USDA NASS': 'USDA NASS',
    'Investing.com': 'Investing.com',
    'AGMARKNET': 'AGMARKNET'
}


# =============================================================================
# ETL PIPELINE CLASS
# =============================================================================
class ETLPipeline:
    """
    ETL Pipeline for agricultural market data.
    
    Processes raw JSON files and loads data into SQLite database.
    """
    
    def __init__(
        self,
        input_dir: str = None,
        db_path: str = None
    ):
        """
        Initialize the ETL pipeline.
        
        Args:
            input_dir: Directory containing raw JSON files
            db_path: Path to SQLite database
        """
        self.input_dir = Path(input_dir or config.DATA_DIR)
        self.db_path = db_path or config.DATABASE_PATH
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'records_extracted': 0,
            'records_transformed': 0,
            'records_loaded': 0,
            'duplicates_removed': 0,
            'anomalies_detected': 0
        }
    
    # -------------------------------------------------------------------------
    # EXTRACT PHASE
    # -------------------------------------------------------------------------
    def extract(self, file_pattern: str = "*.json") -> List[Dict]:
        """
        Extract data from raw JSON files.
        
        Args:
            file_pattern: Glob pattern for input files
            
        Returns:
            List of all extracted records
        """
        logger.info(f"Extracting data from {self.input_dir}")
        
        all_records = []
        json_files = list(self.input_dir.glob(file_pattern))
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Case 1: Direct list of records
                if isinstance(data, list):
                    all_records.extend(data)
                
                # Case 2: Dict of lists (keys are commodities)
                elif isinstance(data, dict):
                    # Check if it has 'commodities' wrapper (old structure/USDA)
                    if 'commodities' in data:
                        commodities_data = data['commodities']
                        source = data.get('source', 'Unknown')
                        if isinstance(commodities_data, dict):
                             for comm, recs in commodities_data.items():
                                 if isinstance(recs, list):
                                     for r in recs:
                                         r['source'] = r.get('source', source)
                                         all_records.append(r)
                    
                    # Check if it's a direct mapping (Investing/Agmarknet)
                    else:
                        for key, value in data.items():
                            if isinstance(value, list):
                                for r in value:
                                    if isinstance(r, dict):
                                        all_records.append(r)
                
                self.stats['files_processed'] += 1
                logger.info(f"Extracted from {json_file.name}")
                
            except Exception as e:
                logger.error(f"Error extracting {json_file}: {e}")
        
        self.stats['records_extracted'] = len(all_records)
        logger.info(f"Extracted {len(all_records)} total records")
        
        return all_records
    
    # -------------------------------------------------------------------------
    # TRANSFORM PHASE
    # -------------------------------------------------------------------------
    def transform(self, records: List[Dict]) -> pd.DataFrame:
        """
        Transform extracted records.
        
        Args:
            records: List of raw records
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming data...")
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        initial_count = len(df)
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Ensure required columns exist
        required_cols = ['commodity', 'date', 'source']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Normalize price column
        if 'price' not in df.columns:
            df['price'] = np.nan
            
        # Fill missing prices from modal_price if available (though modal_price won't be saved)
        if 'modal_price' in df.columns:
            df['price'] = df['price'].fillna(df['modal_price'])
            
        if 'high' in df.columns and 'low' in df.columns:
            df['price'] = df['price'].fillna((df['high'] + df['low']) / 2)
            
        # Convert prices to numeric
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df.dropna(subset=['price'])
            df = df[df['price'] > 0]
        
        # Normalize units
        if 'unit' not in df.columns:
            df['unit'] = 'Unknown'
            
        # Apply unit conversions
        for unit, conversion in PRICE_UNIT_CONVERSIONS.items():
            mask = df['unit'] == unit
            if mask.any():
                df.loc[mask, 'price'] *= conversion['factor']
                df.loc[mask, 'unit'] = conversion['target']
        
        # Normalize source names
        df['source'] = df['source'].map(lambda x: SOURCE_MAPPINGS.get(x, x))
        
        # Remove duplicates based on strict schema columns
        before_dedupe = len(df)
        df = df.drop_duplicates(
            subset=['commodity', 'date', 'source'],
            keep='first'
        )
        self.stats['duplicates_removed'] = before_dedupe - len(df)
        
        self.stats['records_transformed'] = len(df)
        logger.info(f"Transformed: {initial_count} -> {len(df)} records")
        logger.info(f"Removed {self.stats['duplicates_removed']} duplicates")
        
        return df
    
    # -------------------------------------------------------------------------
    # LOAD PHASE
    # -------------------------------------------------------------------------
    def load(self, df: pd.DataFrame) -> int:
        """
        Load transformed data into SQLite database.
        
        Args:
            df: Transformed DataFrame
            
        Returns:
            Number of records loaded
        """
        if df.empty:
            logger.warning("No data to load")
            return 0
        
        logger.info(f"Loading data to {self.db_path}")
        
        # Create database directory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        
        try:
            # Create schema
            self._create_schema(conn)
            
            # Get or create lookups
            commodity_map = self._get_commodity_map(conn, df['commodity'].unique())
            source_map = self._get_source_map(conn, df['source'].unique())
            
            # Insert price data
            records_loaded = 0
            for _, row in df.iterrows():
                try:
                    commodity_id = commodity_map.get(row['commodity'])
                    source_id = source_map.get(row['source'])
                    
                    if commodity_id and source_id:
                        conn.execute("""
                            INSERT INTO price_data 
                            (commodity_id, source_id, price, unit, date)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            commodity_id,
                            source_id,
                            row.get('price'),
                            row.get('unit', 'Unknown'),
                            row['date'].strftime('%Y-%m-%d')
                        ))
                        records_loaded += 1
                        
                except Exception as e:
                    logger.debug(f"Error inserting record: {e}")
            
            conn.commit()
            
            # Create daily aggregates
            self._create_aggregates(conn)
            
            self.stats['records_loaded'] = records_loaded
            logger.info(f"Loaded {records_loaded} records")
            
            return records_loaded
            
        finally:
            conn.close()
    
    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema if not exists."""
        schema_file = PROJECT_ROOT / "sql" / "schema.sql"
        
        if schema_file.exists():
            with open(schema_file) as f:
                conn.executescript(f.read())
        else:
            # Minimal schema
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS commodities (
                    commodity_id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    category TEXT
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    url TEXT,
                    reliability_score REAL DEFAULT 8.0
                );
                CREATE TABLE IF NOT EXISTS price_data (
                    price_id INTEGER PRIMARY KEY,
                    commodity_id INTEGER,
                    source_id INTEGER,
                    price REAL,
                    unit TEXT,
                    date DATE,
                    FOREIGN KEY (commodity_id) REFERENCES commodities(commodity_id),
                    FOREIGN KEY (source_id) REFERENCES sources(source_id)
                );
                CREATE TABLE IF NOT EXISTS daily_aggregates (
                    commodity_id INTEGER,
                    date DATE,
                    avg_price REAL,
                    min REAL,
                    max REAL,
                    std_dev REAL,
                    FOREIGN KEY (commodity_id) REFERENCES commodities(commodity_id),
                    UNIQUE (commodity_id, date)
                );
            """)
    
    def _get_commodity_map(
        self,
        conn: sqlite3.Connection,
        commodities: List[str]
    ) -> Dict[str, int]:
        """Get or create commodity mappings."""
        mapping = {}
        
        for commodity in commodities:
            # Get existing
            cursor = conn.execute(
                "SELECT commodity_id FROM commodities WHERE name = ?",
                (commodity,)
            )
            row = cursor.fetchone()
            
            if row:
                mapping[commodity] = row[0]
            else:
                # Insert new
                cursor = conn.execute(
                    "INSERT INTO commodities (name, category) VALUES (?, ?)",
                    (commodity, self._guess_category(commodity))
                )
                mapping[commodity] = cursor.lastrowid
                conn.commit()
        
        return mapping
    
    def _get_source_map(
        self,
        conn: sqlite3.Connection,
        sources: List[str]
    ) -> Dict[str, int]:
        """Get or create source mappings."""
        mapping = {}
        
        source_urls = {
            'USDA NASS': 'https://quickstats.nass.usda.gov',
            'Investing.com': 'https://www.investing.com/commodities/agricultural',
            'AGMARKNET': 'https://agmarknet.gov.in'
        }
        
        for source in sources:
            cursor = conn.execute(
                "SELECT source_id FROM sources WHERE name = ?",
                (source,)
            )
            row = cursor.fetchone()
            
            if row:
                mapping[source] = row[0]
            else:
                cursor = conn.execute(
                    "INSERT INTO sources (name, url, reliability_score) VALUES (?, ?, ?)",
                    (source, source_urls.get(source, ''), 8.0)
                )
                mapping[source] = cursor.lastrowid
                conn.commit()
        
        return mapping
    
    def _guess_category(self, commodity: str) -> str:
        """Guess commodity category."""
        cereals = ['wheat', 'rice', 'corn', 'maize', 'barley', 'oats']
        vegetables = ['tomato', 'onion', 'potato', 'carrot']
        oils = ['soybeans', 'sunflower', 'mustard']
        beverages = ['coffee', 'tea', 'cocoa']
        fibers = ['cotton']
        sugars = ['sugar', 'sugarcane']
        
        name_lower = commodity.lower()
        
        if name_lower in cereals:
            return 'Cereals'
        elif name_lower in vegetables:
            return 'Vegetables'
        elif name_lower in oils:
            return 'Oilseeds'
        elif name_lower in beverages:
            return 'Beverages'
        elif name_lower in fibers:
            return 'Fibers'
        elif name_lower in sugars:
            return 'Sugars'
        else:
            return 'Other'
    
    def _create_aggregates(self, conn: sqlite3.Connection) -> None:
        """Create daily price aggregates."""
        conn.execute("DELETE FROM daily_aggregates")
        
        conn.execute("""
            INSERT INTO daily_aggregates 
            (commodity_id, date, avg_price, min, max, std_dev)
            SELECT 
                commodity_id,
                date,
                AVG(price),
                MIN(price),
                MAX(price),
                0
            FROM price_data
            GROUP BY commodity_id, date
        """)
        
        conn.commit()
        logger.info("Created daily aggregates")
    
    # -------------------------------------------------------------------------
    # RUN PIPELINE
    # -------------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline.
        
        Returns:
            Pipeline statistics
        """
        logger.info("=" * 60)
        logger.info("Starting ETL Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Extract
        records = self.extract()
        
        # Transform
        df = self.transform(records)
        
        # Load
        self.load(df)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['duration_seconds'] = round(duration, 2)
        
        logger.info("=" * 60)
        logger.info("ETL Pipeline Complete")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Loaded: {self.stats['records_loaded']} records")
        logger.info("=" * 60)
        
        return self.stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point for ETL pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='INNOFarms ETL Pipeline')
    parser.add_argument('--input', type=str, default=None,
                        help='Input directory for raw JSON files')
    parser.add_argument('--db', type=str, default=None,
                        help='Output database path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INNOFarms ETL Pipeline")
    print("=" * 60)
    
    pipeline = ETLPipeline(
        input_dir=args.input,
        db_path=args.db
    )
    
    stats = pipeline.run()
    
    print("\nPipeline Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
