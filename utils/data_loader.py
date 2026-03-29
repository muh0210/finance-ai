"""
data_loader.py — Stock Data Fetcher & Cleaner

PURPOSE:
    This module connects to Yahoo Finance via the yfinance API and pulls
    historical stock data. It handles:
    - Fetching OHLCV (Open, High, Low, Close, Volume) data
    - Cleaning missing values
    - Caching data to avoid redundant API calls
    - Validating stock symbols

WHY yfinance?
    - Free, no API key needed
    - Covers all major exchanges (NYSE, NASDAQ, BSE, NSE, etc.)
    - Returns clean pandas DataFrames
    - Supports crypto, ETFs, and indices too
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Cache directory — saves fetched data locally so we don't spam the API
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def validate_symbol(symbol: str) -> bool:
    """
    Check if a stock symbol is valid by attempting a small data fetch.
    
    Returns True if the symbol exists on Yahoo Finance, False otherwise.
    This prevents the app from crashing on bad user input.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        # If the ticker has no market cap or price, it's likely invalid
        return info.get("regularMarketPrice") is not None or info.get("currentPrice") is not None
    except Exception:
        return False


def get_stock_info(symbol: str) -> dict:
    """
    Fetch metadata about a stock (company name, sector, market cap, etc.)
    
    This is used in the UI to show context about what the user is analyzing.
    Clients love seeing "Apple Inc. | Technology | $2.8T Market Cap"
    instead of just "AAPL".
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "description": info.get("longBusinessSummary", "No description available."),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", None),
        }
    except Exception:
        return {"name": symbol, "sector": "N/A", "industry": "N/A", "market_cap": 0}


def load_data(symbol: str = "AAPL", period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters:
    -----------
    symbol : str
        Stock ticker symbol (e.g., "AAPL", "GOOGL", "TSLA")
    period : str
        How far back to fetch. Options: "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
        We default to 2 years — enough data for ML without going too far back
        (market conditions change, old data can hurt predictions).
    use_cache : bool
        If True, check local cache first before hitting the API.
    
    Returns:
    --------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex (the trading dates)
    
    HOW IT WORKS:
    1. Check if we have cached data that's fresh (< 1 hour old)
    2. If yes → load from CSV (instant, no API call)
    3. If no → fetch from yfinance, clean it, save to cache
    """
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{period}.csv")
    
    # --- Try loading from cache ---
    if use_cache and os.path.exists(cache_file):
        file_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
        if file_age_hours < 1:  # Cache is fresh (less than 1 hour old)
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not data.empty:
                    return data
            except Exception:
                pass  # Cache corrupted, fetch fresh
    
    # --- Fetch fresh data from Yahoo Finance ---
    data = yf.download(symbol, period=period, progress=False)
    
    if data.empty:
        raise ValueError(f"No data found for symbol '{symbol}'. Check if the ticker is correct.")
    
    # --- Clean the data ---
    # Drop any rows with missing values (market holidays, gaps)
    data.dropna(inplace=True)
    
    # Handle MultiIndex columns that yfinance sometimes returns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Keep only the columns we need
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [col for col in required_cols if col in data.columns]
    data = data[available_cols]
    
    # --- Save to cache ---
    os.makedirs(CACHE_DIR, exist_ok=True)
    data.to_csv(cache_file)
    
    return data


def load_multiple(symbols: list, period: str = "2y") -> dict:
    """
    Fetch data for multiple stocks at once.
    Used for portfolio comparison and multi-stock analysis.
    
    Returns a dict: { "AAPL": DataFrame, "GOOGL": DataFrame, ... }
    """
    result = {}
    for symbol in symbols:
        try:
            result[symbol] = load_data(symbol, period)
        except ValueError:
            continue  # Skip invalid symbols
    return result
