"""
backtest.py — Historical Backtesting Engine

PURPOSE:
    Answer the question every client asks:
    "If I had followed this model's advice, would I have made money?"

    This module simulates trading on HISTORICAL data using the model's
    predictions, then compares performance against simply buying and holding.

HOW IT WORKS:
    1. Split data into rolling windows
    2. At each step: train on past data, predict next day
    3. If model says UP → we hold/buy the stock
    4. If model says DOWN → we go to cash (sell)
    5. Track cumulative returns over time
    6. Compare to buy-and-hold strategy

WHY THIS MATTERS:
    - Model accuracy of 55% means NOTHING without context
    - Backtesting shows REAL dollar performance
    - A model that's only 52% accurate can still be profitable
      if it catches big moves and avoids big drops
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils.indicators import FEATURE_COLUMNS


def run_backtest(data: pd.DataFrame, lookback: int = 200, step: int = 1) -> dict:
    """
    Walk-forward backtest of the ML model.
    
    WALK-FORWARD METHOD:
        Day 200: Train on days 1-199, predict day 200
        Day 201: Train on days 1-200, predict day 201
        Day 202: Train on days 1-201, predict day 202
        ... and so on
        
        This ensures we NEVER use future data (no look-ahead bias).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with all indicators already computed
    lookback : int
        Minimum training window size (default 200 days)
    step : int
        Re-train every N days (1 = every day, 5 = every week)
        Higher step = faster backtest but less accurate
    
    Returns:
    --------
    dict with:
        - dates: list of dates
        - strategy_returns: cumulative returns of our model's strategy
        - buyhold_returns: cumulative returns of buy-and-hold
        - signals: list of predictions (0/1) at each step
        - accuracy: overall prediction accuracy
        - total_return_strategy: final return %
        - total_return_buyhold: final return %
        - sharpe_ratio: risk-adjusted return metric
        - max_drawdown: worst peak-to-trough decline
        - win_rate: % of profitable trades
        - trades: total number of position changes
    """
    available_features = [f for f in FEATURE_COLUMNS if f in data.columns]
    
    if not available_features:
        raise ValueError("No feature columns found in data. Run add_all_indicators() first.")
    
    # Create target
    data = data.copy()
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data.dropna(subset=["Target"] + available_features, inplace=True)
    
    if len(data) < lookback + 50:
        raise ValueError(f"Not enough data for backtesting. Need {lookback + 50}, have {len(data)}.")
    
    dates = []
    predictions = []
    actuals = []
    strategy_daily_returns = []
    buyhold_daily_returns = []
    
    # Walk-forward loop
    for i in range(lookback, len(data) - 1, step):
        # Training data: everything before current point
        train_data = data.iloc[:i]
        # Test point: current day
        test_point = data.iloc[i:i+1]
        
        X_train = train_data[available_features]
        y_train = train_data["Target"]
        X_test = test_point[available_features]
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        
        # Predict
        pred = model.predict(X_test)[0]
        actual = data["Target"].iloc[i]
        
        # Calculate daily return
        daily_return = data["Return"].iloc[i + 1] if i + 1 < len(data) else 0
        
        # Strategy return: if we predicted UP, we hold the stock (get the return)
        # If we predicted DOWN, we're in cash (0% return)
        strategy_return = daily_return if pred == 1 else 0
        
        dates.append(data.index[i])
        predictions.append(int(pred))
        actuals.append(int(actual))
        strategy_daily_returns.append(strategy_return)
        buyhold_daily_returns.append(daily_return)
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + pd.Series(strategy_daily_returns)).cumprod()
    buyhold_cumulative = (1 + pd.Series(buyhold_daily_returns)).cumprod()
    
    # Calculate metrics
    accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(predictions)
    
    # Sharpe Ratio (annualized)
    strategy_series = pd.Series(strategy_daily_returns)
    if strategy_series.std() > 0:
        sharpe = (strategy_series.mean() / strategy_series.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max Drawdown
    cummax = strategy_cumulative.cummax()
    drawdown = np.where(
        cummax > 0,
        (strategy_cumulative - cummax) / cummax,
        0
    )
    max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
    
    # Win Rate (of days we were in the market)
    in_market_returns = [r for r, p in zip(strategy_daily_returns, predictions) if p == 1]
    if len(in_market_returns) > 0:
        win_rate = sum(1 for r in in_market_returns if r > 0) / len(in_market_returns)
    else:
        win_rate = 0.0
    
    # Trade count (position changes)
    trades = sum(1 for i in range(1, len(predictions)) if predictions[i] != predictions[i-1])
    
    # Sortino Ratio
    downside_returns = strategy_series[strategy_series < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = (strategy_series.mean() / downside_returns.std()) * np.sqrt(252)
    else:
        sortino = 0
    
    return {
        "dates": dates,
        "strategy_returns": strategy_cumulative.tolist(),
        "buyhold_returns": buyhold_cumulative.tolist(),
        "signals": predictions,
        "actuals": actuals,
        "accuracy": accuracy,
        "total_return_strategy": (strategy_cumulative.iloc[-1] - 1) * 100,
        "total_return_buyhold": (buyhold_cumulative.iloc[-1] - 1) * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown * 100,
        "win_rate": win_rate,
        "trades": trades,
        "days_tested": len(predictions),
    }


def calculate_support_resistance(data: pd.DataFrame, window: int = 20, num_levels: int = 3) -> dict:
    """
    Find Support and Resistance levels.
    
    WHAT ARE THEY?
        Support: Price level where the stock tends to STOP FALLING (floor)
        Resistance: Price level where the stock tends to STOP RISING (ceiling)
    
    HOW WE FIND THEM:
        1. Find local minima (support) and maxima (resistance)
        2. Cluster nearby levels together
        3. Return the strongest ones (most touches)
    
    Parameters:
    -----------
    data : pd.DataFrame with High, Low, Close columns
    window : int, lookback window for finding peaks/troughs
    num_levels : int, number of support/resistance levels to return
    
    Returns:
    --------
    dict with 'support' and 'resistance' lists of price levels
    """
    prices = data["Close"].values
    highs = data["High"].values
    lows = data["Low"].values
    
    supports = []
    resistances = []
    
    # Find local minima (support) and maxima (resistance)
    for i in range(window, len(prices) - window):
        # Local minimum → support
        if lows[i] == min(lows[i - window:i + window + 1]):
            supports.append(lows[i])
        
        # Local maximum → resistance
        if highs[i] == max(highs[i - window:i + window + 1]):
            resistances.append(highs[i])
    
    # Cluster nearby levels (within 1.5% of each other)
    def cluster_levels(levels, threshold=0.015):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for level in levels[1:]:
            if (level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        # Return the average of each cluster, sorted by cluster size (strength)
        result = [(np.mean(c), len(c)) for c in clusters]
        result.sort(key=lambda x: -x[1])  # Most touches first
        return result[:num_levels]
    
    current_price = prices[-1]
    
    # Only keep supports below current price and resistances above
    support_levels = cluster_levels([s for s in supports if s < current_price])
    resistance_levels = cluster_levels([r for r in resistances if r > current_price])
    
    return {
        "support": [{"price": round(p, 2), "strength": s} for p, s in support_levels],
        "resistance": [{"price": round(p, 2), "strength": s} for p, s in resistance_levels],
        "current_price": round(current_price, 2),
    }


def calculate_advanced_metrics(data: pd.DataFrame) -> dict:
    """
    Calculate advanced financial metrics that real analysts use.
    
    SHARPE RATIO:
        Risk-adjusted return. Higher = better.
        > 1.0 = Good, > 2.0 = Very Good, > 3.0 = Excellent
        Formula: (Mean Return - Risk-Free Rate) / Std Dev of Returns
        We use 0 as risk-free rate for simplicity.
    
    SORTINO RATIO:
        Like Sharpe but only penalizes DOWNSIDE volatility.
        A stock that goes up wildly isn't really "risky" — only drops are.
    
    CALMAR RATIO:
        Return / Max Drawdown. How much pain for how much gain?
    
    BETA:
        How much the stock moves relative to the market.
        Beta > 1 = More volatile than market
        Beta < 1 = Less volatile than market
    """
    returns = data["Return"].dropna()
    
    # Annualized return
    total_return = (data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1
    trading_days = len(data)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1
    
    # Annualized volatility
    annualized_vol = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (risk-free rate ≈ 0 for simplicity)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    
    # Sortino Ratio
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = annualized_return / downside_vol if downside_vol > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdowns.min())
    
    # Calmar Ratio
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
    
    # Value at Risk (95% confidence)
    var_95 = returns.quantile(0.05)
    
    # Average daily gain vs average daily loss
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    avg_gain = gains.mean() if len(gains) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = abs(avg_gain * len(gains)) / abs(avg_loss * len(losses)) if len(losses) > 0 and avg_loss != 0 else 0
    
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "profit_factor": profit_factor,
        "positive_days": len(gains),
        "negative_days": len(losses),
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "avg_daily_gain": avg_gain,
        "avg_daily_loss": avg_loss,
    }
