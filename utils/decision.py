"""
decision.py — Decision Engine + Explainer + Risk Scorer

PURPOSE:
    This is the BRAIN of the system — the "product" layer.
    It takes raw ML output and turns it into:
    1. A clear BUY / SELL / HOLD decision
    2. A risk score (Low / Medium / High / Very High)
    3. A human-readable explanation of WHY

WHY THIS MODULE IS CRITICAL:
    The ML model outputs a number (0 or 1). That's useless to a human.
    This module translates numbers into ACTIONABLE ADVICE with REASONING.
    
    Clients don't care about "prediction = 1, confidence = 0.67".
    They care about:
    "BUY — The stock is oversold (RSI 28), momentum is turning positive,
     and volatility is low. Confidence: 67%. Risk: Medium."
    
    THAT'S what makes this premium.
"""

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1. DECISION ENGINE — Buy / Sell / Hold
# ═══════════════════════════════════════════════════════════════════════════

def make_decision(prediction: int, confidence: float, rsi: float, 
                  macd_hist: float, volatility: float, bb_position: float) -> dict:
    """
    Generate a BUY / SELL / HOLD decision using multiple signals.
    
    DECISION LOGIC (multi-factor):
    
    We don't rely on just ONE signal. We combine:
    - ML prediction (UP/DOWN)
    - Confidence level (how sure the model is)
    - RSI (overbought/oversold)
    - MACD histogram (momentum direction)
    - Bollinger Band position (price extremes)
    
    STRONG BUY:  ML says UP + RSI < 30 (oversold) + MACD positive
    BUY:         ML says UP + reasonable RSI
    HOLD:        Low confidence OR mixed signals
    SELL:        ML says DOWN + reasonable RSI
    STRONG SELL: ML says DOWN + RSI > 70 (overbought) + MACD negative
    
    Parameters:
    -----------
    prediction : int (0=DOWN, 1=UP)
    confidence : float (0.0 to 1.0)
    rsi : float (0 to 100)
    macd_hist : float (positive = bullish momentum, negative = bearish)
    volatility : float (daily return std dev)
    bb_position : float (0 = at lower band, 1 = at upper band)
    
    Returns:
    --------
    dict with 'action', 'strength', 'emoji'
    """
    # Calculate a composite score (-100 to +100)
    score = 0
    
    # ML prediction contributes most (weighted by confidence)
    if prediction == 1:
        score += 40 * confidence
    else:
        score -= 40 * confidence
    
    # RSI contribution
    if rsi < 30:
        score += 20  # Oversold = bullish signal
    elif rsi < 40:
        score += 10
    elif rsi > 70:
        score -= 20  # Overbought = bearish signal
    elif rsi > 60:
        score -= 10
    
    # MACD momentum
    if macd_hist > 0:
        score += 15
    else:
        score -= 15
    
    # Bollinger Band position
    if bb_position < 0.2:
        score += 10  # Near lower band = potential bounce
    elif bb_position > 0.8:
        score -= 10  # Near upper band = potential pullback
    
    # Volatility penalty (high volatility = less certainty)
    if volatility > 0.03:
        score *= 0.7  # Reduce conviction in volatile markets
    
    # Map score to decision
    if score > 30:
        return {"action": "STRONG BUY", "strength": min(score, 100), "emoji": "🟢🔥"}
    elif score > 10:
        return {"action": "BUY", "strength": score, "emoji": "🟢"}
    elif score > -10:
        return {"action": "HOLD", "strength": abs(score), "emoji": "🟡"}
    elif score > -30:
        return {"action": "SELL", "strength": abs(score), "emoji": "🔴"}
    else:
        return {"action": "STRONG SELL", "strength": min(abs(score), 100), "emoji": "🔴🔥"}


# ═══════════════════════════════════════════════════════════════════════════
# 2. RISK SCORING — Low / Medium / High / Very High
# ═══════════════════════════════════════════════════════════════════════════

def calculate_risk(data: pd.DataFrame) -> dict:
    """
    Multi-factor risk assessment.
    
    RISK FACTORS:
    1. Volatility (biggest factor) — How wildly the price swings
    2. Maximum drawdown — Worst peak-to-trough decline
    3. Volume stability — Erratic volume = uncertainty
    4. Price trend consistency — Choppy vs smooth trends
    
    RISK LEVELS:
    - Low (0-25):      Stable stock, predictable movement
    - Medium (26-50):   Normal market conditions
    - High (51-75):     Elevated uncertainty, caution advised
    - Very High (76-100): Extreme volatility, high chance of big loss
    
    Returns dict with score (0-100), level, color, and factors breakdown.
    """
    risk_score = 0
    factors = []
    
    # --- Factor 1: Current Volatility (0-35 points) ---
    current_vol = data["Volatility"].iloc[-1] if "Volatility" in data.columns else 0
    vol_score = min(current_vol / 0.05 * 35, 35)  # 5% daily vol = max score
    risk_score += vol_score
    
    if current_vol > 0.03:
        factors.append(("High daily volatility", "⚠️", "dangerous"))
    elif current_vol > 0.015:
        factors.append(("Moderate volatility", "⚡", "warning"))
    else:
        factors.append(("Low volatility", "✅", "safe"))
    
    # --- Factor 2: Maximum Drawdown (0-25 points) ---
    if "Close" in data.columns:
        rolling_max = data["Close"].rolling(window=50).max()
        drawdown = (data["Close"] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        dd_score = min(max_drawdown / 0.3 * 25, 25)  # 30% drawdown = max
        risk_score += dd_score
        
        if max_drawdown > 0.2:
            factors.append((f"Significant drawdown ({max_drawdown:.1%})", "📉", "dangerous"))
        elif max_drawdown > 0.1:
            factors.append((f"Moderate drawdown ({max_drawdown:.1%})", "📊", "warning"))
        else:
            factors.append((f"Contained drawdown ({max_drawdown:.1%})", "✅", "safe"))
    
    # --- Factor 3: Volume Instability (0-20 points) ---
    if "Volume_ratio" in data.columns:
        vol_std = data["Volume_ratio"].tail(20).std()
        vol_instability_score = min(vol_std / 1.0 * 20, 20)
        risk_score += vol_instability_score
        
        if vol_std > 0.8:
            factors.append(("Erratic trading volume", "📊", "warning"))
        else:
            factors.append(("Stable trading volume", "✅", "safe"))
    
    # --- Factor 4: RSI Extreme (0-20 points) ---
    if "RSI" in data.columns:
        rsi = data["RSI"].iloc[-1]
        if rsi > 80 or rsi < 20:
            risk_score += 20
            factors.append(("RSI at extreme level", "⚠️", "dangerous"))
        elif rsi > 70 or rsi < 30:
            risk_score += 10
            factors.append(("RSI approaching extreme", "⚡", "warning"))
        else:
            factors.append(("RSI in normal range", "✅", "safe"))
    
    # Clamp to 0-100
    risk_score = min(max(risk_score, 0), 100)
    
    # Determine level and color
    if risk_score <= 25:
        level, color = "Low", "#22c55e"       # Green
    elif risk_score <= 50:
        level, color = "Medium", "#f59e0b"     # Amber
    elif risk_score <= 75:
        level, color = "High", "#ef4444"       # Red
    else:
        level, color = "Very High", "#dc2626"  # Dark Red
    
    return {
        "score": round(risk_score, 1),
        "level": level,
        "color": color,
        "factors": factors,
        "volatility": float(current_vol) if not pd.isna(current_vol) else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. EXPLANATION ENGINE — Human-readable reasoning
# ═══════════════════════════════════════════════════════════════════════════

def generate_explanation(data: pd.DataFrame, prediction: int, confidence: float,
                         feature_importance: pd.Series, decision: dict) -> list:
    """
    Generate a detailed, human-readable explanation of WHY the system
    made its decision.
    
    THIS IS THE PREMIUM FEATURE.
    
    Instead of "Model says buy", we say:
    "The model recommends BUYING because:
     1. RSI is at 28 (oversold territory) — historically, stocks bounce from here
     2. MACD momentum is turning positive — trend reversal likely
     3. The stock is near the bottom of its Bollinger Band — room to move up
     4. The model is 73% confident based on 200 similar historical patterns"
    
    Parameters:
    -----------
    data : pd.DataFrame with indicators
    prediction : 0 or 1
    confidence : 0.0 to 1.0
    feature_importance : Series with feature name → importance score
    decision : dict from make_decision()
    
    Returns:
    --------
    List of explanation strings, each one a clear reason.
    """
    explanations = []
    latest = data.iloc[-1]
    
    # --- 1. Overall direction ---
    direction = "UPWARD" if prediction == 1 else "DOWNWARD"
    explanations.append(
        f"📈 The ML model predicts **{direction}** movement with "
        f"**{confidence:.0%}** confidence, based on analysis of "
        f"{len(data)} trading days of historical data."
    )
    
    # --- 2. RSI Analysis ---
    rsi = latest.get("RSI", 50)
    if rsi > 70:
        explanations.append(
            f"⚠️ **RSI is {rsi:.1f}** (above 70 = OVERBOUGHT). The stock has risen "
            f"aggressively and may be due for a pullback. Buyers should exercise caution."
        )
    elif rsi < 30:
        explanations.append(
            f"💡 **RSI is {rsi:.1f}** (below 30 = OVERSOLD). The stock has been heavily "
            f"sold — this often presents a buying opportunity as prices tend to bounce back."
        )
    elif rsi > 60:
        explanations.append(
            f"📊 **RSI is {rsi:.1f}** (moderately high). The stock shows bullish momentum "
            f"but isn't overextended yet."
        )
    elif rsi < 40:
        explanations.append(
            f"📊 **RSI is {rsi:.1f}** (moderately low). The stock shows bearish pressure "
            f"but hasn't reached oversold levels."
        )
    else:
        explanations.append(
            f"📊 **RSI is {rsi:.1f}** (neutral zone 40-60). No overbought or oversold "
            f"condition — the market is balanced."
        )
    
    # --- 3. MACD Analysis ---
    macd_hist = latest.get("MACD_hist", 0)
    if macd_hist > 0:
        explanations.append(
            f"🟢 **MACD histogram is positive** ({macd_hist:.4f}). "
            f"Bullish momentum is building — the fast EMA is above the slow EMA."
        )
    else:
        explanations.append(
            f"🔴 **MACD histogram is negative** ({macd_hist:.4f}). "
            f"Bearish momentum — the fast EMA has crossed below the slow EMA."
        )
    
    # --- 4. Bollinger Band Position ---
    bb_pos = latest.get("BB_position", 0.5)
    if bb_pos > 0.8:
        explanations.append(
            f"📈 Price is near the **upper Bollinger Band** (position: {bb_pos:.2f}). "
            f"The stock is trading at the high end of its recent range — potential resistance ahead."
        )
    elif bb_pos < 0.2:
        explanations.append(
            f"📉 Price is near the **lower Bollinger Band** (position: {bb_pos:.2f}). "
            f"The stock is trading at the low end of its recent range — potential support/bounce."
        )
    
    # --- 5. Volatility Context ---
    vol = latest.get("Volatility", 0)
    if vol > 0.03:
        explanations.append(
            f"⚡ **High volatility** detected ({vol:.4f}). "
            f"Large price swings are likely. Any prediction carries more uncertainty."
        )
    elif vol < 0.01:
        explanations.append(
            f"✅ **Low volatility** ({vol:.4f}). "
            f"The stock is moving calmly, which increases prediction reliability."
        )
    
    # --- 6. Moving Average Trend ---
    ma10_dist = latest.get("MA10_dist", 0)
    ma50_dist = latest.get("MA50_dist", 0)
    if ma10_dist > 0 and ma50_dist > 0:
        explanations.append(
            f"📈 Price is **above both MA10 and MA50** — confirming an uptrend."
        )
    elif ma10_dist < 0 and ma50_dist < 0:
        explanations.append(
            f"📉 Price is **below both MA10 and MA50** — confirming a downtrend."
        )
    elif ma10_dist > 0 and ma50_dist < 0:
        explanations.append(
            f"🔄 **Mixed signals**: Price is above short-term MA10 but below "
            f"medium-term MA50 — potential trend reversal in progress."
        )
    
    # --- 7. Top Features ---
    if feature_importance is not None and len(feature_importance) > 0:
        top_3 = feature_importance.head(3)
        feature_desc = {
            "RSI": "momentum strength",
            "MACD": "trend momentum",
            "MACD_hist": "momentum direction",
            "Volatility": "price stability",
            "Volatility_20": "longer-term volatility",
            "MA10_dist": "short-term trend position",
            "MA50_dist": "medium-term trend position",
            "BB_position": "Bollinger Band position",
            "Return_5d": "weekly price momentum",
            "Return_20d": "monthly price momentum",
            "Volume_ratio": "trading activity",
            "Price_position_20": "20-day price range position",
            "Price_position_50": "50-day price range position",
        }
        
        top_names = [f"**{name}** ({feature_desc.get(name, name)})" for name in top_3.index]
        explanations.append(
            f"🧠 The model's top decision drivers were: {', '.join(top_names)}."
        )
    
    return explanations


# ═══════════════════════════════════════════════════════════════════════════
# 4. PORTFOLIO OPTIMIZATION (BONUS)
# ═══════════════════════════════════════════════════════════════════════════

def simple_portfolio_suggestion(analyses: dict) -> list:
    """
    Given analysis results for multiple stocks, suggest allocation.
    
    Uses a simple scoring system:
    - BUY signals get positive weight
    - SELL signals get negative weight
    - Weight by confidence and inverse of risk
    
    Parameters:
    -----------
    analyses : dict of { symbol: { "decision": {...}, "risk": {...}, "confidence": float } }
    
    Returns:
    --------
    List of dicts: [{"symbol": "AAPL", "allocation": 0.45, "reason": "..."}, ...]
    """
    scores = {}
    
    for symbol, analysis in analyses.items():
        action = analysis["decision"]["action"]
        confidence = analysis.get("confidence", 0.5)
        risk_score = analysis["risk"]["score"]
        
        # Base score from action
        action_scores = {
            "STRONG BUY": 2.0,
            "BUY": 1.0,
            "HOLD": 0.0,
            "SELL": -1.0,
            "STRONG SELL": -2.0,
        }
        
        base = action_scores.get(action, 0)
        # Weight by confidence and penalize by risk
        risk_penalty = 1 - (risk_score / 100) * 0.5  # High risk = lower allocation
        score = base * confidence * risk_penalty
        
        if score > 0:
            scores[symbol] = score
    
    if not scores:
        return [{"symbol": "Cash", "allocation": 1.0, "reason": "No attractive opportunities found. Stay in cash."}]
    
    # Normalize to percentages
    total = sum(scores.values())
    suggestions = []
    for symbol, score in sorted(scores.items(), key=lambda x: -x[1]):
        alloc = score / total
        suggestions.append({
            "symbol": symbol,
            "allocation": round(alloc, 4),
            "reason": f"Score: {score:.2f} (Action: {analyses[symbol]['decision']['action']}, "
                      f"Risk: {analyses[symbol]['risk']['level']})"
        })
    
    return suggestions
