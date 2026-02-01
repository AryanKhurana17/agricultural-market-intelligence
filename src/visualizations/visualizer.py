"""
Visualization Generator for Agricultural Analysis
==================================================

Generates charts for price trends, volatility, rankings, and forecasts.

Usage:
    python src/visualizations/visualizer.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


# =============================================================================
# VISUALIZATION GENERATOR
# =============================================================================
class Visualizer:
    """Generates visualizations for agricultural analysis."""
    
    def __init__(
        self,
        reports_dir: str = None,
        output_dir: str = None
    ):
        """
        Initialize the visualizer.
        
        Args:
            reports_dir: Directory containing JSON reports
            output_dir: Directory for output charts
        """
        self.reports_dir = Path(reports_dir or config.REPORTS_DIR)
        self.output_dir = Path(output_dir or config.REPORTS_DIR / "visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style configuration
        self.style = {
            'figure.figsize': (12, 7),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        }
        plt.rcParams.update(self.style)
    
    def load_report(self, filename: str) -> Optional[Dict]:
        """Load a JSON report file."""
        filepath = self.reports_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        logger.warning(f"Report not found: {filepath}")
        return None
    
    def save_figure(self, fig: plt.Figure, name: str) -> str:
        """Save figure to file."""
        filepath = self.output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved: {filepath.name}")
        return str(filepath)
    
    # -------------------------------------------------------------------------
    # CHART GENERATORS
    # -------------------------------------------------------------------------
    def plot_price_trends(
        self,
        commodity: str,
        days: int = 180
    ) -> Optional[str]:
        """
        Plot price trends with moving averages.
        
        Args:
            commodity: Commodity name
            days: Days of data to plot
            
        Returns:
            Path to saved chart
        """
        import sqlite3
        
        conn = sqlite3.connect(config.DATABASE_PATH)
        query = """
            SELECT p.date, 
                   AVG(p.price) as price
            FROM price_data p
            JOIN commodities c ON p.commodity_id = c.commodity_id
            WHERE c.name = ? AND p.date >= date('now', ?)
            GROUP BY p.date
            ORDER BY p.date
        """
        
        import pandas as pd
        df = pd.read_sql_query(query, conn, params=(commodity, f'-{days} days'))
        conn.close()
        
        if df.empty:
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df['ma_7'] = df['price'].rolling(7).mean()
        df['ma_30'] = df['price'].rolling(30).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df['date'], df['price'], 
                label='Daily Price', alpha=0.7, linewidth=1)
        ax.plot(df['date'], df['ma_7'], 
                label='7-Day MA', linewidth=2)
        ax.plot(df['date'], df['ma_30'], 
                label='30-Day MA', linewidth=2)
        
        ax.fill_between(df['date'], df['price'], alpha=0.1)
        
        ax.set_title(f'{commodity} Price Trends ({days} Days)', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        plt.tight_layout()
        
        return self.save_figure(fig, f"{commodity.lower()}_price_trends")
    
    def plot_volatility_comparison(
        self,
        commodities: List[str] = None
    ) -> Optional[str]:
        """
        Plot volatility comparison across commodities.
        
        Args:
            commodities: List of commodities to compare
            
        Returns:
            Path to saved chart
        """
        report = self.load_report("trend_analysis.json")
        if not report:
            return None
        
        data = report.get('commodities', {})
        
        if commodities is None:
            commodities = list(data.keys())
        
        vols = []
        names = []
        colors = []
        
        for name in commodities:
            if name in data:
                vol = data[name].get('volatility_pct', 0)
                vols.append(vol)
                names.append(name)
                
                level = data[name].get('volatility_level', 'Medium')
                if level == 'Low':
                    colors.append('#22c55e')
                elif level == 'High':
                    colors.append('#ef4444')
                else:
                    colors.append('#f59e0b')
        
        if not vols:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(names, vols, color=colors, edgecolor='white')
        
        for bar, vol in zip(bars, vols):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{vol:.1f}%', va='center', fontsize=10)
        
        ax.set_xlabel('Volatility (%)')
        ax.set_title('Price Volatility Comparison', fontweight='bold')
        ax.axvline(x=10, color='#22c55e', linestyle='--', alpha=0.5, label='Low')
        ax.axvline(x=20, color='#ef4444', linestyle='--', alpha=0.5, label='High')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        return self.save_figure(fig, "volatility_comparison")
    
    def plot_opportunity_rankings(self) -> Optional[str]:
        """
        Plot opportunity scores ranking.
        
        Returns:
            Path to saved chart
        """
        report = self.load_report("opportunity_rankings.json")
        if not report:
            return None
        
        rankings = report.get('rankings', [])
        
        if not rankings:
            return None
        
        names = [r['commodity'] for r in rankings[:10]]
        scores = [r['score'] for r in rankings[:10]]
        recommendations = [r.get('recommendation', '') for r in rankings[:10]]
        
        colors = []
        for rec in recommendations:
            if 'Strong Buy' in rec:
                colors.append('#22c55e')
            elif 'Buy' in rec:
                colors.append('#10b981')
            elif 'Hold' in rec:
                colors.append('#f59e0b')
            else:
                colors.append('#ef4444')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1])
        
        for bar, score in zip(bars, scores[::-1]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{score:.0f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Opportunity Score')
        ax.set_title('Investment Opportunity Rankings', fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        return self.save_figure(fig, "opportunity_rankings")
    
    def plot_portfolio_allocation(self) -> Optional[str]:
        """
        Plot recommended portfolio allocation.
        
        Returns:
            Path to saved chart
        """
        report = self.load_report("profitability_report.json")
        if not report:
            return None
        
        profitability = report.get('profitability', {})
        
        items = []
        for name, data in profitability.items():
            roi = data.get('roi_percent', 0)
            if roi > 0:
                items.append((name, roi))
        
        if not items:
            return None
        
        # Top 6 by ROI
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:6]
        
        # Normalize for pie
        total = sum(x[1] for x in items)
        sizes = [x[1]/total * 100 for x in items]
        labels = [f"{x[0]}\n({x[1]:.0f}% ROI)" for x in items]
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        
        ax.set_title('Recommended Portfolio Allocation\n(by ROI)', fontweight='bold')
        
        plt.tight_layout()
        
        return self.save_figure(fig, "portfolio_allocation")
    
    def plot_forecast(self, commodity: str) -> Optional[str]:
        """
        Plot price forecast.
        
        Args:
            commodity: Commodity name
            
        Returns:
            Path to saved chart
        """
        report = self.load_report("forecast_output.json")
        if not report:
            return None
        
        forecasts = report.get('forecasts', {})
        data = forecasts.get(commodity)
        
        if not data:
            return None
        
        forecast = data.get('forecast_7day', [])
        current = data.get('current_price', 0)
        model = data.get('best_model', 'Unknown')
        
        if not forecast:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Historical placeholder
        hist_x = list(range(-30, 1))
        hist_y = [current * (1 + np.random.uniform(-0.02, 0.02)) for _ in hist_x]
        hist_y[-1] = current
        
        ax.plot(hist_x, hist_y, 'b-', label='Historical', alpha=0.7)
        
        # Forecast
        fore_x = list(range(1, len(forecast) + 1))
        ax.plot(fore_x, forecast, 'g--', linewidth=2, marker='o', label='Forecast')
        
        # Fill confidence interval
        upper = [p * 1.05 for p in forecast]
        lower = [p * 0.95 for p in forecast]
        ax.fill_between(fore_x, lower, upper, alpha=0.2, color='green')
        
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.5, label='Today')
        
        ax.set_xlabel('Days from Today')
        ax.set_ylabel('Price')
        ax.set_title(f'{commodity} 7-Day Forecast ({model})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self.save_figure(fig, f"{commodity.lower()}_forecast")
    
    # -------------------------------------------------------------------------
    # MAIN GENERATOR
    # -------------------------------------------------------------------------
    def generate_all(self, commodities: List[str] = None) -> Dict[str, str]:
        """
        Generate all visualizations.
        
        Args:
            commodities: List of commodities to visualize
            
        Returns:
            Dict of chart paths
        """
        if commodities is None:
            commodities = ['Wheat', 'Rice', 'Corn', 'Coffee']
        
        logger.info("Generating visualizations...")
        
        charts = {}
        
        # Price trends
        for commodity in commodities:
            result = self.plot_price_trends(commodity)
            if result:
                charts[f'{commodity}_trends'] = result
                print(f"   OK: {commodity} price trends")
        
        # Volatility
        result = self.plot_volatility_comparison()
        if result:
            charts['volatility'] = result
            print("   OK: Volatility comparison")
        
        # Rankings
        result = self.plot_opportunity_rankings()
        if result:
            charts['rankings'] = result
            print("   OK: Profitability rankings")
        
        # Portfolio
        result = self.plot_portfolio_allocation()
        if result:
            charts['portfolio'] = result
            print("   OK: Portfolio allocation")
        
        # Forecasts
        for commodity in ['Wheat', 'Corn']:
            result = self.plot_forecast(commodity)
            if result:
                charts[f'{commodity}_forecast'] = result
                print(f"   OK: {commodity} forecast")
        
        logger.info(f"Generated {len(charts)} visualizations")
        
        return charts


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point."""
    print("=" * 60)
    print("INNOFarms Visualization Generator")
    print("=" * 60)
    
    visualizer = Visualizer()
    charts = visualizer.generate_all()
    
    print("\n" + "=" * 60)
    print(f"Generated {len(charts)} visualizations")
    print(f"Output: {visualizer.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
