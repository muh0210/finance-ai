"""
indicators.py — Technical Indicator Engine

PURPOSE:
    Transform raw price data into meaningful FEATURES that the ML model
    can learn from. Raw price alone is useless — the model needs patterns.

WHY FEATURE ENGINEERING MATTERS:
    Imagine you're a doctor. A patient's temperature alone means little.
    But temperature + blood pressure + heart rate + symptoms = diagnosis.
    
    Same here:
    - Price alone = useless
    - Price + RSI + MACD + Volatility + Volume trends = actionable signal

INDICATORS WE COMPUTE:
    1. Moving Averages (MA10, MA50, MA200) — Trend direction
    2. RSI (Relative Strength Index) — Overbought/Oversold
    3. MACD — Momentum & trend changes
    4. Bollinger Bands — Volatility channels
    5. Daily Returns — Day-to-day price changes
    6. Volatility — Risk measurement
    7. Volume Change — Trading activity shifts
    8. Price Position — Where price sits relative to its range
"""

import pandas as pd
import numpy as np


def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """
    Moving Averages — THE most fundamental indicator.
    
    WHAT IT DOES:
        Smooths out price noise to reveal the underlying trend.
    
    HOW TO READ:
        - Price ABOVE MA → Uptrend (bullish)
        - Price BELOW MA → Downtrend (bearish)
        - MA10 crosses ABOVE MA50 → "Golden Cross" (strong buy signal)
        - MA10 crosses BELOW MA50 → "Death Cross" (strong sell signal)
    
    WHY 3 DIFFERENT PERIODS (10, 50, 200)?
        - MA10 : Short-term trend (reacts fast, noisy)
        - MA50 : Medium-term trend (balanced)
        - MA200: Long-term trend (slow, reliable)
    """
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA200"] = data["Close"].rolling(window=200).mean()
    
    # --- Derived features the model loves ---
    # How far is the current price from the moving average? (in %)
    data["MA10_dist"] = (data["Close"] - data["MA10"]) / data["MA10"] * 100
    data["MA50_dist"] = (data["Close"] - data["MA50"]) / data["MA50"] * 100
    
    return data


def add_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI — Relative Strength Index (0 to 100)
    
    WHAT IT MEASURES:
        The SPEED and MAGNITUDE of recent price changes.
        Tells you if a stock is being bought too aggressively (overbought)
        or sold too aggressively (oversold).
    
    HOW TO READ:
        RSI > 70 → OVERBOUGHT: Stock has risen too fast. Likely to pull back.
                    Think of it like a rubber band stretched too far — it snaps back.
        RSI < 30 → OVERSOLD: Stock has fallen too hard. Likely to bounce up.
                    Bargain hunters start buying.
        RSI 40-60 → NEUTRAL: No extreme condition.
    
    THE MATH:
        1. Calculate daily price changes (up days vs down days)
        2. Average the gains and losses over 14 days
        3. RS = Average Gain / Average Loss
        4. RSI = 100 - (100 / (1 + RS))
    
    WHY 14 DAYS?
        Industry standard set by J. Welles Wilder (inventor of RSI).
        14 days ≈ 2 trading weeks. Short enough to be responsive,
        long enough to filter noise.
    """
    delta = data["Close"].diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].fillna(50)  # Default to neutral if incalculable
    
    return data


def add_macd(data: pd.DataFrame) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence
    
    WHAT IT DOES:
        Shows the RELATIONSHIP between two moving averages of a stock's price.
        It reveals momentum shifts BEFORE they happen in price.
    
    COMPONENTS:
        - MACD Line = EMA(12) - EMA(26)    (fast EMA minus slow EMA)
        - Signal Line = EMA(9) of MACD Line (smoothed version)
        - Histogram = MACD - Signal         (the gap between them)
    
    HOW TO READ:
        - MACD crosses ABOVE Signal → Bullish (momentum turning up)
        - MACD crosses BELOW Signal → Bearish (momentum turning down)
        - Histogram growing → Trend is strengthening
        - Histogram shrinking → Trend is weakening
    
    WHY EMA (Exponential) INSTEAD OF SMA (Simple)?
        EMA gives MORE WEIGHT to recent prices → reacts faster to changes.
        In trading, recent data matters more than old data.
    """
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    
    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]
    
    return data


def add_bollinger_bands(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Bollinger Bands — Volatility-based price channels
    
    WHAT IT DOES:
        Creates an "envelope" around the price:
        - Middle Band = 20-day SMA
        - Upper Band = SMA + 2 standard deviations
        - Lower Band = SMA - 2 standard deviations
    
    HOW TO READ:
        - Price touches UPPER band → Potentially overbought
        - Price touches LOWER band → Potentially oversold
        - Bands SQUEEZE (narrow) → Low volatility, BIG move coming
        - Bands EXPAND (wide) → High volatility, moving fast
    
    WHY 2 STANDARD DEVIATIONS?
        Statistically, ~95% of price action stays within 2 SDs.
        When price breaks out, it's SIGNIFICANT.
    """
    sma = data["Close"].rolling(window=period).mean()
    std = data["Close"].rolling(window=period).std()
    
    data["BB_upper"] = sma + (std * 2)
    data["BB_lower"] = sma - (std * 2)
    data["BB_middle"] = sma
    
    # How close is the price to the upper/lower band? (0 = lower, 1 = upper)
    bb_range = data["BB_upper"] - data["BB_lower"]
    data["BB_position"] = np.where(
        bb_range > 0,
        (data["Close"] - data["BB_lower"]) / bb_range,
        0.5  # Default to midpoint when bands are flat
    )
    
    return data


def add_volatility_and_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns and Volatility — The foundation of risk measurement.
    
    DAILY RETURN:
        How much the stock moved today vs yesterday (in %).
        Formula: (Today's Close - Yesterday's Close) / Yesterday's Close
    
    VOLATILITY:
        The standard deviation of returns over a window.
        HIGH volatility = stock is jumping around wildly = RISKY
        LOW volatility = stock is moving smoothly = SAFER
    
    WHY THIS MATTERS:
        - A stock that goes up 2% every day → Low volatility, great!
        - A stock that goes +10%, -8%, +12%, -9% → Same average return
          but MUCH higher risk. You could lose big on any given day.
    """
    data["Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Return"].rolling(window=10).std()
    data["Volatility_20"] = data["Return"].rolling(window=20).std()
    
    # Cumulative return over the last 5 days (weekly momentum)
    data["Return_5d"] = data["Close"].pct_change(periods=5)
    # Cumulative return over the last 20 days (monthly momentum)
    data["Return_20d"] = data["Close"].pct_change(periods=20)
    
    return data


def add_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Volume Features — Shows CONVICTION behind price moves.
    
    WHY VOLUME MATTERS:
        Price going UP on HIGH volume → Strong move, likely to continue
        Price going UP on LOW volume → Weak move, likely to reverse
        
        Think of it as confidence:
        - If 1 person says "buy" → not convincing
        - If 10,000 people say "buy" → something real is happening
    """
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_ratio"] = data["Volume"] / data["Volume_MA20"]
    
    return data


def add_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Price Position Features — Where does the price sit in its range?
    
    Helps the model understand if we're at a local high or low.
    """
    # Price position within the last 20 days (0 = lowest, 1 = highest)
    range_20 = data["High"].rolling(20).max() - data["Low"].rolling(20).min()
    data["Price_position_20"] = np.where(
        range_20 > 0,
        (data["Close"] - data["Low"].rolling(20).min()) / range_20,
        0.5  # Default to midpoint when range is zero
    )
    
    # Price position within the last 50 days
    range_50 = data["High"].rolling(50).max() - data["Low"].rolling(50).min()
    data["Price_position_50"] = np.where(
        range_50 > 0,
        (data["Close"] - data["Low"].rolling(50).min()) / range_50,
        0.5  # Default to midpoint when range is zero
    )
    
    return data


def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — add ALL indicators in one call.
    
    This is what app.py calls. Clean, simple interface.
    
    Order matters:
    1. Moving Averages first (other indicators may depend on them)
    2. RSI
    3. MACD
    4. Bollinger Bands
    5. Returns & Volatility
    6. Volume
    7. Price features
    8. Drop incomplete rows (indicators need historical data to compute)
    """
    data = add_moving_averages(data)
    data = add_rsi(data)
    data = add_macd(data)
    data = add_bollinger_bands(data)
    data = add_volatility_and_returns(data)
    data = add_volume_features(data)
    data = add_price_features(data)
    
    # Drop rows where indicators couldn't be calculated
    # (first ~200 rows won't have MA200, for example)
    data.dropna(inplace=True)
    
    return data


# ---------------------------------------------------------------------------
# FEATURE LIST — used by the ML model
# These are the columns the model trains on. Adding/removing features here
# changes what the model sees.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "MA10_dist",        # Distance from 10-day MA (%)
    "MA50_dist",        # Distance from 50-day MA (%)
    "RSI",              # Relative Strength Index
    "MACD",             # MACD line value
    "MACD_hist",        # MACD histogram (momentum)
    "BB_position",      # Position within Bollinger Bands
    "Volatility",       # 10-day volatility
    "Volatility_20",    # 20-day volatility
    "Return_5d",        # 5-day return (weekly momentum)
    "Return_20d",       # 20-day return (monthly momentum)
    "Volume_ratio",     # Volume relative to 20-day average
    "Price_position_20",# Price position in 20-day range
    "Price_position_50",# Price position in 50-day range
]
