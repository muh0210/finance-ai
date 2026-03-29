"""
sentiment.py — News Sentiment Analysis Engine

PURPOSE:
    Price moves on NEWS, not just technicals. This module:
    1. Fetches recent news headlines for a stock
    2. Scores each headline as Positive / Negative / Neutral
    3. Calculates an overall sentiment score
    
    Adds a completely new data dimension beyond technical indicators.

HOW SENTIMENT ANALYSIS WORKS:
    We use a keyword-based approach with weighted scoring:
    - Positive words: "surge", "beat", "profit", "upgrade" → +score
    - Negative words: "crash", "miss", "loss", "downgrade" → -score
    - Intensity modifiers: "significantly", "massive" → amplify score
    
    This avoids heavy ML dependencies while still being effective.
    Financial-specific keywords outperform generic sentiment for stocks.

DATA SOURCE:
    Google News RSS feed — free, no API key needed, works globally.
"""

import re
import feedparser
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════
# Financial Sentiment Lexicon
# Curated for stock/financial news — much better than generic sentiment
# ═══════════════════════════════════════════════════════════════════════════

POSITIVE_WORDS = {
    # Strong positive (weight 2)
    "surge": 2, "soar": 2, "skyrocket": 2, "rally": 2, "boom": 2,
    "breakthrough": 2, "record high": 2, "all-time high": 2, "blowout": 2,
    "outperform": 2, "upgrade": 2, "strong buy": 2, "beat expectations": 2,
    
    # Moderate positive (weight 1)
    "rise": 1, "gain": 1, "grow": 1, "profit": 1, "revenue": 1,
    "beat": 1, "exceed": 1, "positive": 1, "bullish": 1, "optimistic": 1,
    "recovery": 1, "rebound": 1, "upbeat": 1, "strong": 1, "robust": 1,
    "expand": 1, "advance": 1, "improve": 1, "dividend": 1, "buyback": 1,
    "innovation": 1, "launch": 1, "partnership": 1, "deal": 1, "approval": 1,
    "outpace": 1, "momentum": 1, "confidence": 1, "upgrade": 1, "buy": 1,
}

NEGATIVE_WORDS = {
    # Strong negative (weight 2)
    "crash": 2, "plunge": 2, "collapse": 2, "tank": 2, "freefall": 2,
    "bankruptcy": 2, "fraud": 2, "scandal": 2, "investigation": 2,
    "downgrade": 2, "sell-off": 2, "selloff": 2, "recession": 2,
    "layoff": 2, "lawsuit": 2, "default": 2,
    
    # Moderate negative (weight 1)
    "fall": 1, "drop": 1, "decline": 1, "loss": 1, "miss": 1,
    "disappoint": 1, "weak": 1, "bearish": 1, "pessimistic": 1,
    "concern": 1, "risk": 1, "volatile": 1, "uncertainty": 1,
    "warning": 1, "cut": 1, "reduce": 1, "struggle": 1, "pressure": 1,
    "debt": 1, "overvalued": 1, "delay": 1, "recall": 1, "fine": 1,
    "regulation": 1, "tariff": 1, "slowdown": 1, "inflation": 1,
}

INTENSITY_MODIFIERS = {
    "significantly": 1.5, "dramatically": 1.5, "sharply": 1.5,
    "massively": 1.8, "huge": 1.5, "major": 1.3, "substantial": 1.3,
    "slight": 0.5, "modest": 0.7, "marginally": 0.5, "slightly": 0.5,
}


def analyze_headline(headline: str) -> dict:
    """
    Score a single news headline for financial sentiment.
    
    Returns:
        dict with 'score' (-1.0 to +1.0), 'label' (Positive/Negative/Neutral),
        'color', and matched 'keywords'
    """
    text = headline.lower()
    score = 0
    matched_keywords = []
    intensity = 1.0
    
    # Check intensity modifiers
    for modifier, mult in INTENSITY_MODIFIERS.items():
        if modifier in text:
            intensity = mult
            break
    
    # Score positive words
    for word, weight in POSITIVE_WORDS.items():
        if word in text:
            score += weight * intensity
            matched_keywords.append(f"+{word}")
    
    # Score negative words
    for word, weight in NEGATIVE_WORDS.items():
        if word in text:
            score -= weight * intensity
            matched_keywords.append(f"-{word}")
    
    # Normalize to -1.0 to +1.0 range
    if score > 0:
        normalized = min(score / 4.0, 1.0)
    elif score < 0:
        normalized = max(score / 4.0, -1.0)
    else:
        normalized = 0
    
    # Determine label and color
    if normalized > 0.15:
        label, color, emoji = "Positive", "#22c55e", "🟢"
    elif normalized < -0.15:
        label, color, emoji = "Negative", "#ef4444", "🔴"
    else:
        label, color, emoji = "Neutral", "#f59e0b", "🟡"
    
    return {
        "headline": headline,
        "score": round(normalized, 3),
        "label": label,
        "color": color,
        "emoji": emoji,
        "keywords": matched_keywords,
    }


def fetch_news(symbol: str, company_name: str = "", max_articles: int = 10) -> list:
    """
    Fetch recent news for a stock from Google News RSS.
    
    Parameters:
    -----------
    symbol : str - Stock ticker (e.g., "AAPL")
    company_name : str - Full company name (e.g., "Apple") for better search
    max_articles : int - Maximum number of articles to fetch
    
    Returns:
    --------
    List of dicts, each with: title, link, published, source
    """
    # Search query: use both ticker and company name for better results
    query = f"{symbol} stock"
    if company_name and company_name != symbol:
        query = f"{company_name} {symbol} stock"
    
    # Google News RSS feed
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(url)
        
        articles = []
        for entry in feed.entries[:max_articles]:
            # Extract source from title (Google News format: "Title - Source")
            title = entry.get("title", "")
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1] if len(parts) > 1 else ""
            
            # Parse published date
            published = entry.get("published", "")
            try:
                pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') and entry.published_parsed else None
            except Exception:
                pub_date = None
            
            articles.append({
                "title": title,
                "link": entry.get("link", ""),
                "published": published,
                "pub_date": pub_date,
                "source": source,
            })
        
        return articles
    
    except Exception as e:
        return []


def get_sentiment_analysis(symbol: str, company_name: str = "") -> dict:
    """
    Full sentiment analysis: fetch news → score each headline → aggregate.
    
    Returns:
    --------
    dict with:
        - articles: list of analyzed articles
        - overall_score: -1.0 to +1.0
        - overall_label: Positive/Negative/Neutral
        - positive_count, negative_count, neutral_count
        - summary: human-readable summary
    """
    articles = fetch_news(symbol, company_name)
    
    if not articles:
        return {
            "articles": [],
            "overall_score": 0,
            "overall_label": "Neutral",
            "overall_color": "#f59e0b",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "summary": "No recent news found for this stock.",
        }
    
    # Analyze each headline
    analyzed = []
    for article in articles:
        sentiment = analyze_headline(article["title"])
        sentiment["link"] = article["link"]
        sentiment["published"] = article["published"]
        sentiment["source"] = article["source"]
        analyzed.append(sentiment)
    
    # Aggregate scores
    scores = [a["score"] for a in analyzed]
    overall_score = sum(scores) / len(scores)
    
    positive_count = sum(1 for a in analyzed if a["label"] == "Positive")
    negative_count = sum(1 for a in analyzed if a["label"] == "Negative")
    neutral_count = sum(1 for a in analyzed if a["label"] == "Neutral")
    
    # Overall label
    if overall_score > 0.1:
        overall_label, overall_color = "Positive", "#22c55e"
    elif overall_score < -0.1:
        overall_label, overall_color = "Negative", "#ef4444"
    else:
        overall_label, overall_color = "Neutral", "#f59e0b"
    
    # Generate summary
    total = len(analyzed)
    summary = (
        f"Analyzed {total} recent news articles. "
        f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral. "
    )
    
    if overall_score > 0.3:
        summary += "News sentiment is strongly bullish — media coverage is overwhelmingly positive."
    elif overall_score > 0.1:
        summary += "News sentiment leans positive — more favorable coverage than negative."
    elif overall_score < -0.3:
        summary += "News sentiment is strongly bearish — significant negative media coverage."
    elif overall_score < -0.1:
        summary += "News sentiment leans negative — concerning headlines outweigh positive ones."
    else:
        summary += "News sentiment is mixed/neutral — no strong directional bias from media."
    
    return {
        "articles": analyzed,
        "overall_score": round(overall_score, 3),
        "overall_label": overall_label,
        "overall_color": overall_color,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "summary": summary,
    }
