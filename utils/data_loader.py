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
import re
from datetime import datetime


# ---------------------------------------------------------------------------
# Cache directory — saves fetched data locally so we don't spam the API
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Valid ticker regex: 1-10 alphanumeric chars, optionally with dots/dashes (BRK.B, BF-B)
_VALID_SYMBOL_RE = re.compile(r"^[A-Za-z0-9\.\-]{1,10}$")


def validate_symbol(symbol: str) -> bool:
    """
    Check if a stock symbol is valid by attempting a small data fetch.

    Returns True if the symbol exists on Yahoo Finance, False otherwise.
    This prevents the app from crashing on bad user input.
    """
    if not symbol or not _VALID_SYMBOL_RE.match(symbol):
        return False
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
    default_info = {
        "name": symbol,
        "sector": "N/A",
        "industry": "N/A",
        "market_cap": 0,
        "currency": "USD",
        "exchange": "N/A",
        "description": "No description available.",
        "52w_high": 0,
        "52w_low": 0,
        "pe_ratio": None,
        "dividend_yield": None,
    }
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "market_cap": info.get("marketCap") or 0,
            "currency": info.get("currency") or "USD",
            "exchange": info.get("exchange") or "N/A",
            "description": info.get("longBusinessSummary") or "No description available.",
            "52w_high": info.get("fiftyTwoWeekHigh") or 0,
            "52w_low": info.get("fiftyTwoWeekLow") or 0,
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
        }
    except Exception:
        return default_info


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle yfinance MultiIndex columns that occur in newer versions.

    yfinance >= 0.2.31 may return columns like:
        MultiIndex([('Close', 'AAPL'), ('High', 'AAPL'), ...])
    or sometimes:
        MultiIndex([('Price', ...), ...])

    This function flattens them to simple column names: 'Close', 'High', etc.
    """
    if isinstance(data.columns, pd.MultiIndex):
        # Try level 0 first (usually has 'Close', 'Open', etc.)
        level0_cols = data.columns.get_level_values(0).tolist()

        # Check if level 0 has the standard column names
        standard_cols = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        if any(col in standard_cols for col in level0_cols):
            data.columns = level0_cols
        else:
            # Some yfinance versions wrap everything under 'Price' at level 0
            # In that case, try level 1
            try:
                level1_cols = data.columns.get_level_values(1).tolist()
                if any(col in standard_cols for col in level1_cols):
                    data.columns = level1_cols
                else:
                    # Last resort: just flatten by joining levels
                    data.columns = [
                        col[0] if isinstance(col, tuple) else col
                        for col in data.columns
                    ]
            except (IndexError, ValueError):
                data.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in data.columns
                ]

        # Remove duplicate column names (e.g. if 'Close' appears twice)
        data = data.loc[:, ~data.columns.duplicated()]

    return data


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

    Raises:
    -------
    ValueError: If symbol is invalid or no data returned.
    """
    # Input validation
    symbol = str(symbol).strip().upper()
    if not _VALID_SYMBOL_RE.match(symbol):
        raise ValueError(f"Invalid stock symbol: '{symbol}'. Use 1-10 alphanumeric characters.")

    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{period}.csv")

    # --- Try loading from cache ---
    if use_cache and os.path.exists(cache_file):
        file_age_hours = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 3600
        if file_age_hours < 1:  # Cache is fresh (less than 1 hour old)
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not data.empty and "Close" in data.columns:
                    return data
            except Exception:
                pass  # Cache corrupted, fetch fresh

    # --- Fetch fresh data from Yahoo Finance ---
    try:
        data = yf.download(symbol, period=period, progress=False)
    except Exception as e:
        raise ValueError(f"Failed to download data for '{symbol}': {e}")

    if data is None or data.empty:
        raise ValueError(f"No data found for symbol '{symbol}'. Check if the ticker is correct.")

    # --- Flatten MultiIndex columns ---
    data = _flatten_columns(data)

    # --- Clean the data ---
    # Drop any rows with missing values (market holidays, gaps)
    data.dropna(inplace=True)

    # Keep only the columns we need
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [col for col in required_cols if col in data.columns]

    if "Close" not in available_cols:
        raise ValueError(
            f"Data for '{symbol}' is missing the 'Close' column. "
            f"Available columns: {list(data.columns)}. "
            f"This may be a yfinance version issue."
        )

    data = data[available_cols]

    if data.empty:
        raise ValueError(f"No valid data rows for '{symbol}' after cleaning.")

    # --- Save to cache ---
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        data.to_csv(cache_file)
    except Exception:
        pass  # Cache write failure is non-critical

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
