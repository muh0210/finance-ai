"""
app.py — AI Financial Decision Assistant (Streamlit UI) v2.0

NEW IN v2.0:
    - Backtesting with equity curve chart
    - News Sentiment Analysis
    - PDF Report Download
    - Support/Resistance Levels
    - Advanced Financial Metrics (Sharpe, Sortino, Calmar)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
warnings.filterwarnings("ignore")

# Our modules
from utils.data_loader import load_data, get_stock_info
from utils.indicators import add_all_indicators, FEATURE_COLUMNS
from utils.model import train_model
from utils.decision import make_decision, calculate_risk, generate_explanation, simple_portfolio_suggestion
from utils.backtest import run_backtest, calculate_support_resistance, calculate_advanced_metrics
from utils.sentiment import get_sentiment_analysis
from utils.report import generate_report


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Financial Decision Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium dark theme
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* --- Global --- */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* --- Main cards --- */
    .metric-card {
        background: linear-gradient(145deg, rgba(30,30,60,0.8), rgba(20,20,40,0.9));
        border: 1px solid rgba(100,100,255,0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .metric-card h3 {
        color: #a0a0ff;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* --- Decision badge --- */
    .decision-buy {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 2px solid #10b981;
        color: #34d399;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 0 30px rgba(16,185,129,0.2);
    }
    
    .decision-sell {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
        color: #fca5a5;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 0 30px rgba(239,68,68,0.2);
    }
    
    .decision-hold {
        background: linear-gradient(135deg, #78350f, #92400e);
        border: 2px solid #f59e0b;
        color: #fcd34d;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 0 30px rgba(245,158,11,0.2);
    }
    
    .decision-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        margin-top: 8px;
        opacity: 0.9;
    }
    
    /* --- Explanation cards --- */
    .explanation-card {
        background: rgba(30,30,60,0.6);
        border-left: 4px solid #6366f1;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 10px 0;
        color: #e0e0ff;
        line-height: 1.6;
    }
    
    /* --- Risk meter --- */
    .risk-meter {
        background: rgba(20,20,40,0.8);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(100,100,255,0.15);
    }
    
    /* --- Disclaimer --- */
    .disclaimer {
        background: rgba(50,20,20,0.4);
        border: 1px solid rgba(255,100,100,0.2);
        border-radius: 12px;
        padding: 16px;
        color: #ff9999;
        font-size: 0.85rem;
        margin-top: 20px;
    }
    
    /* --- Section headers --- */
    .section-header {
        color: #c0c0ff;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(100,100,255,0.2);
    }
    
    /* --- Sidebar styling --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* --- News cards --- */
    .news-positive { border-left: 4px solid #22c55e; }
    .news-negative { border-left: 4px solid #ef4444; }
    .news-neutral { border-left: 4px solid #f59e0b; }
    
    .news-card {
        background: rgba(30,30,60,0.5);
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 6px 0;
        color: #d0d0ff;
        font-size: 0.95rem;
    }
    
    /* --- Support/Resistance --- */
    .sr-level {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        margin: 4px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* --- Hide Streamlit branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 AI Financial Assistant")
    st.markdown("---")
    
    # Stock input
    symbol = st.text_input(
        "📌 Stock Symbol",
        value="AAPL",
        help="Enter a stock ticker (e.g., AAPL, GOOGL, TSLA, MSFT, AMZN)"
    ).upper().strip()
    
    # Time period
    period = st.selectbox(
        "📅 Analysis Period",
        options=["1y", "2y", "5y"],
        index=1,
        help="How far back to look. 2 years balances data quantity with relevance."
    )
    
    st.markdown("---")
    
    # Feature toggles
    st.markdown("### ⚙️ Analysis Options")
    run_backtest_opt = st.checkbox("📉 Run Backtest", value=True, help="Test model on historical data")
    run_sentiment_opt = st.checkbox("📰 News Sentiment", value=True, help="Analyze recent news headlines")
    
    st.markdown("---")
    
    # Multi-stock comparison
    st.markdown("### 📊 Multi-Stock Comparison")
    compare_symbols = st.text_input(
        "Compare with (comma-separated)",
        value="",
        placeholder="GOOGL, MSFT, TSLA",
        help="Optional: Compare your stock against others"
    )
    
    st.markdown("---")
    
    # Analyze button
    analyze = st.button("🚀 Analyze Stock", use_container_width=True, type="primary")
    
    # Validate symbol format
    _valid_symbol = bool(re.match(r"^[A-Za-z0-9\.\-]{1,10}$", symbol))
    if not _valid_symbol and symbol:
        st.warning("⚠️ Invalid symbol format. Use 1-10 alphanumeric characters (e.g., AAPL, BRK.B).")
    
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational and 
        informational purposes only. It does NOT constitute financial advice.
        Always consult a qualified financial advisor before making 
        investment decisions.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def format_market_cap(cap):
    """Format large numbers: 2800000000 → $2.80T"""
    if cap >= 1e12:
        return f"${cap/1e12:.2f}T"
    elif cap >= 1e9:
        return f"${cap/1e9:.2f}B"
    elif cap >= 1e6:
        return f"${cap/1e6:.2f}M"
    return f"${cap:,.0f}"


def create_price_chart(data, symbol):
    """Create an interactive candlestick chart with indicators."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.45, 0.18, 0.18, 0.18],
        subplot_titles=(
            f"{symbol} Price & Bollinger Bands",
            "RSI (Relative Strength Index)",
            "MACD (Momentum)",
            "Volume"
        )
    )
    
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data["MA10"], name="MA10",
        line=dict(color="#60a5fa", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50",
        line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
    
    if "BB_upper" in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_upper"], name="BB Upper",
            line=dict(color="rgba(139,92,246,0.4)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_lower"], name="BB Lower",
            line=dict(color="rgba(139,92,246,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(139,92,246,0.05)"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI",
        line=dict(color="#8b5cf6", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)",
                  annotation_text="Overbought (70)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.5)",
                  annotation_text="Oversold (30)", row=2, col=1)
    
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in data["MACD_hist"]]
    fig.add_trace(go.Bar(x=data.index, y=data["MACD_hist"], name="MACD Histogram",
        marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD",
        line=dict(color="#60a5fa", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal",
        line=dict(color="#f59e0b", width=1.2)), row=3, col=1)
    
    vol_colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(data["Close"], data["Open"])]
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.7), row=4, col=1)
    
    fig.update_layout(
        height=800, template="plotly_dark",
        paper_bgcolor="rgba(15,15,26,0.0)", plot_bgcolor="rgba(20,20,40,0.5)",
        font=dict(color="#c0c0ff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False, margin=dict(t=60, b=30, l=60, r=30),
    )
    fig.update_xaxes(gridcolor="rgba(100,100,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(100,100,255,0.1)")
    return fig


def create_feature_importance_chart(importance):
    fig = go.Figure()
    imp_sorted = importance.sort_values(ascending=True)
    colors = [f"rgba(99, 102, 241, {0.4 + 0.6 * (i/len(imp_sorted))})" for i in range(len(imp_sorted))]
    fig.add_trace(go.Bar(x=imp_sorted.values, y=imp_sorted.index, orientation="h",
        marker_color=colors, text=[f"{v:.1%}" for v in imp_sorted.values], textposition="auto"))
    fig.update_layout(title="🧠 What the Model Cares About Most", template="plotly_dark",
        paper_bgcolor="rgba(15,15,26,0.0)", plot_bgcolor="rgba(20,20,40,0.5)",
        font=dict(color="#c0c0ff"), height=400, margin=dict(t=60, b=30, l=120, r=30),
        xaxis_title="Importance Score")
    return fig


def create_backtest_chart(bt_result):
    """Create backtest equity curve chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_result["dates"], y=bt_result["strategy_returns"],
        name="AI Strategy", line=dict(color="#8b5cf6", width=2.5),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.1)"
    ))
    fig.add_trace(go.Scatter(
        x=bt_result["dates"], y=bt_result["buyhold_returns"],
        name="Buy & Hold", line=dict(color="#f59e0b", width=2, dash="dash")
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Break Even")
    fig.update_layout(
        title="📉 Backtest: AI Strategy vs Buy & Hold",
        template="plotly_dark",
        paper_bgcolor="rgba(15,15,26,0.0)", plot_bgcolor="rgba(20,20,40,0.5)",
        font=dict(color="#c0c0ff"), height=400,
        margin=dict(t=60, b=30, l=60, r=30),
        yaxis_title="Portfolio Value ($1 invested)",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="color: #e0e0ff; font-size: 2.5rem; font-weight: 800; 
               background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin-bottom: 5px;">
        🧠 AI Financial Decision Assistant
    </h1>
    <p style="color: #8080b0; font-size: 1.1rem;">
        ML-Powered Stock Analysis · Backtesting · News Sentiment · PDF Reports
    </p>
</div>
""", unsafe_allow_html=True)


if analyze and _valid_symbol:
    try:
        # ─── STEP 1: Fetch Data ───
        with st.spinner(f"📡 Fetching {symbol} data from Yahoo Finance..."):
            data = load_data(symbol, period)
            info = get_stock_info(symbol)
        
        # ─── STEP 2: Stock Info Header ───
        st.markdown(f'<div class="section-header">📋 {info["name"]} ({symbol})</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        current_price = data["Close"].iloc[-1]
        prev_price = data["Close"].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Price</h3>
                <div class="value">${current_price:.2f}</div>
                <div style="color: {'#22c55e' if price_change >= 0 else '#ef4444'}; font-size: 0.95rem;">
                    {'▲' if price_change >= 0 else '▼'} {abs(price_change):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><h3>Market Cap</h3>
                <div class="value">{format_market_cap(info.get('market_cap', 0))}</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><h3>Sector</h3>
                <div class="value" style="font-size: 1.2rem;">{info.get('sector', 'N/A')}</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card"><h3>52W High</h3>
                <div class="value">${info.get('52w_high', 0):.2f}</div></div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""<div class="metric-card"><h3>52W Low</h3>
                <div class="value">${info.get('52w_low', 0):.2f}</div></div>""", unsafe_allow_html=True)
        
        # ─── STEP 3: Add Indicators ───
        with st.spinner("📊 Computing technical indicators..."):
            data = add_all_indicators(data)
        
        # ─── STEP 4: Train Model ───
        with st.spinner("🤖 Training ML model..."):
            result = train_model(data)
        
        # ─── STEP 5: Generate Decision ───
        latest = data.iloc[-1]
        decision = make_decision(
            prediction=result["prediction"], confidence=result["confidence"],
            rsi=latest.get("RSI", 50), macd_hist=latest.get("MACD_hist", 0),
            volatility=latest.get("Volatility", 0.02), bb_position=latest.get("BB_position", 0.5)
        )
        risk = calculate_risk(data)
        explanations = generate_explanation(
            data=data, prediction=result["prediction"], confidence=result["confidence"],
            feature_importance=result["feature_importance"], decision=decision
        )
        
        # ─── STEP 6: Display Decision ───
        st.markdown('<div class="section-header">🎯 AI Decision</div>', unsafe_allow_html=True)
        col_dec, col_risk = st.columns([2, 1])
        
        with col_dec:
            action = decision["action"]
            css_class = "decision-buy" if "BUY" in action else ("decision-sell" if "SELL" in action else "decision-hold")
            st.markdown(f"""
            <div class="{css_class}">
                <p class="decision-text">{decision['emoji']} {action}</p>
                <p class="confidence-text">
                    Model Confidence: {result['confidence']:.0%} | 
                    Model Accuracy: {result['accuracy']:.0%} |
                    CV Accuracy: {result['cv_accuracy']:.0%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk:
            st.markdown(f"""
            <div class="risk-meter" style="border-left: 4px solid {risk['color']};">
                <h3 style="color: #a0a0ff; font-size: 0.85rem; text-transform: uppercase; 
                           letter-spacing: 1.5px;">Risk Assessment</h3>
                <div style="font-size: 2.5rem; font-weight: 700; color: {risk['color']};">
                    {risk['score']:.0f}/100
                </div>
                <div style="font-size: 1.2rem; color: {risk['color']}; font-weight: 600;">
                    {risk['level']} Risk
                </div>
                <div style="margin-top: 12px;">
            """, unsafe_allow_html=True)
            for factor_text, emoji, level in risk["factors"]:
                color = {"safe": "#22c55e", "warning": "#f59e0b", "dangerous": "#ef4444"}.get(level, "#888")
                st.markdown(f'<span style="color: {color}; font-size: 0.9rem;">{emoji} {factor_text}</span><br>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # ─── STEP 7: Explanation ───
        st.markdown('<div class="section-header">💡 Why This Decision? (Explainability)</div>', unsafe_allow_html=True)
        for exp in explanations:
            st.markdown(f'<div class="explanation-card">{exp}</div>', unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # NEW: SUPPORT & RESISTANCE + ADVANCED METRICS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">📏 Support & Resistance Levels</div>', unsafe_allow_html=True)
        
        sr_levels = calculate_support_resistance(data)
        advanced = calculate_advanced_metrics(data)
        
        col_sr, col_adv = st.columns([1, 1])
        
        with col_sr:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Price: ${sr_levels['current_price']:.2f}</h3>
                <div style="margin-top: 12px;">
                    <div style="color: #ef4444; font-weight: 600; margin-bottom: 8px;">🔴 Resistance (Ceiling)</div>
            """, unsafe_allow_html=True)
            for r in sr_levels.get("resistance", []):
                st.markdown(f'<span class="sr-level" style="background: rgba(239,68,68,0.15); color: #fca5a5;">${r["price"]:.2f} ({r["strength"]} touches)</span>', unsafe_allow_html=True)
            st.markdown(f"""
                    <div style="color: #22c55e; font-weight: 600; margin: 12px 0 8px 0;">🟢 Support (Floor)</div>
            """, unsafe_allow_html=True)
            for s in sr_levels.get("support", []):
                st.markdown(f'<span class="sr-level" style="background: rgba(34,197,94,0.15); color: #86efac;">${s["price"]:.2f} ({s["strength"]} touches)</span>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with col_adv:
            sharpe_color = "#22c55e" if advanced["sharpe_ratio"] > 1 else ("#f59e0b" if advanced["sharpe_ratio"] > 0 else "#ef4444")
            st.markdown(f"""
            <div class="metric-card">
                <h3>Advanced Metrics</h3>
                <table style="width: 100%; color: #c0c0ff; font-size: 0.95rem;">
                    <tr><td>📈 Annualized Return</td><td style="text-align:right; font-weight:600;">{advanced['annualized_return']:.2%}</td></tr>
                    <tr><td>📊 Annualized Volatility</td><td style="text-align:right;">{advanced['annualized_volatility']:.2%}</td></tr>
                    <tr><td>⭐ Sharpe Ratio</td><td style="text-align:right; font-weight:600; color:{sharpe_color};">{advanced['sharpe_ratio']:.2f}</td></tr>
                    <tr><td>🎯 Sortino Ratio</td><td style="text-align:right;">{advanced['sortino_ratio']:.2f}</td></tr>
                    <tr><td>📉 Max Drawdown</td><td style="text-align:right; color:#ef4444;">{advanced['max_drawdown']:.2%}</td></tr>
                    <tr><td>⚖️ Profit Factor</td><td style="text-align:right;">{advanced['profit_factor']:.2f}</td></tr>
                    <tr><td>🟢 Best Day</td><td style="text-align:right; color:#22c55e;">{advanced['best_day']:.2%}</td></tr>
                    <tr><td>🔴 Worst Day</td><td style="text-align:right; color:#ef4444;">{advanced['worst_day']:.2%}</td></tr>
                    <tr><td>📊 VaR (95%)</td><td style="text-align:right;">{advanced['var_95']:.2%}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # NEW: NEWS SENTIMENT
        # ═══════════════════════════════════════════════════════════════════
        sentiment_data = None
        if run_sentiment_opt:
            st.markdown('<div class="section-header">📰 News Sentiment Analysis</div>', unsafe_allow_html=True)
            
            with st.spinner("📰 Analyzing news sentiment..."):
                try:
                    sentiment_data = get_sentiment_analysis(symbol, info.get("name", ""))
                except Exception as e:
                    sentiment_data = None
                    st.warning(f"⚠️ Sentiment analysis failed: {e}")
            
            
            # Sentiment overview — only display if we got data
            if sentiment_data and sentiment_data.get("articles") is not None:
                col_sent_score, col_sent_detail = st.columns([1, 2])
                
                with col_sent_score:
                    overall_label = sentiment_data.get("overall_label", "Neutral")
                    score_emoji = "🟢" if overall_label == "Positive" else ("🔴" if overall_label == "Negative" else "🟡")
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center;">
                        <h3>Overall Sentiment</h3>
                        <div style="font-size: 3rem;">{score_emoji}</div>
                        <div class="value" style="color: {sentiment_data.get('overall_color', '#f59e0b')};">
                            {overall_label}
                        </div>
                        <div style="color: #8080b0; margin-top: 8px;">
                            Score: {sentiment_data.get('overall_score', 0):+.2f} | 
                            {sentiment_data.get('positive_count', 0)}🟢 {sentiment_data.get('negative_count', 0)}🔴 {sentiment_data.get('neutral_count', 0)}🟡
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_sent_detail:
                    st.markdown(f'<div class="explanation-card">{sentiment_data.get("summary", "No summary available.")}</div>', unsafe_allow_html=True)
                    
                    for article in sentiment_data.get("articles", [])[:6]:
                        css = f"news-{article.get('label', 'neutral').lower()}"
                        keywords_str = " ".join(article.get("keywords", []))
                        st.markdown(f"""
                        <div class="news-card {css}">
                            {article.get('emoji', '')} <strong>{article.get('headline', '')[:100]}{'...' if len(article.get('headline', '')) > 100 else ''}</strong>
                            <span style="float:right; color: {article.get('color', '#888')}; font-weight: 600;">{article.get('score', 0):+.2f}</span>
                            <br><span style="font-size: 0.8rem; color: #8080a0;">{article.get('source', '')} {keywords_str}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        # ─── Technical Analysis Chart ───
        st.markdown('<div class="section-header">📈 Technical Analysis Chart</div>', unsafe_allow_html=True)
        chart_days = st.slider("Chart range (trading days)", 30, len(data), min(120, len(data)))
        chart_data = data.tail(chart_days)
        st.plotly_chart(create_price_chart(chart_data, symbol), use_container_width=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # NEW: BACKTESTING
        # ═══════════════════════════════════════════════════════════════════
        bt_result = None
        if run_backtest_opt:
            st.markdown('<div class="section-header">📉 Backtesting — Historical Performance</div>', unsafe_allow_html=True)
            
            with st.spinner("⏳ Running walk-forward backtest (this may take a moment)..."):
                try:
                    bt_result = run_backtest(data, lookback=200, step=3)
                    
                    # Equity curve chart
                    st.plotly_chart(create_backtest_chart(bt_result), use_container_width=True)
                    
                    # Metrics
                    bt_cols = st.columns(6)
                    strategy_color = "#22c55e" if bt_result["total_return_strategy"] > 0 else "#ef4444"
                    buyhold_color = "#22c55e" if bt_result["total_return_buyhold"] > 0 else "#ef4444"
                    
                    metrics = [
                        ("AI Strategy Return", f"{bt_result['total_return_strategy']:+.1f}%", strategy_color),
                        ("Buy & Hold Return", f"{bt_result['total_return_buyhold']:+.1f}%", buyhold_color),
                        ("Backtest Accuracy", f"{bt_result['accuracy']:.0%}", "#c0c0ff"),
                        ("Sharpe Ratio", f"{bt_result['sharpe_ratio']:.2f}", "#c0c0ff"),
                        ("Win Rate", f"{bt_result['win_rate']:.0%}", "#c0c0ff"),
                        ("Max Drawdown", f"{bt_result['max_drawdown']:.1f}%", "#ef4444"),
                    ]
                    
                    for i, (name, value, color) in enumerate(metrics):
                        with bt_cols[i]:
                            st.markdown(f"""
                            <div class="metric-card" style="text-align: center;">
                                <h3>{name}</h3>
                                <div class="value" style="font-size: 1.4rem; color: {color};">{value}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Interpretation
                    if bt_result["total_return_strategy"] > bt_result["total_return_buyhold"]:
                        st.markdown("""
                        <div class="explanation-card" style="border-left-color: #22c55e;">
                            ✅ <strong>The AI strategy outperformed buy-and-hold</strong> on historical data. 
                            The model successfully avoided some downturns by going to cash during predicted down days.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="explanation-card" style="border-left-color: #f59e0b;">
                            ⚠️ <strong>Buy-and-hold outperformed the AI strategy</strong> on historical data. 
                            This can happen in strong bull markets where staying invested beats any switching strategy.
                            The AI may still add value through risk reduction (lower drawdowns).
                        </div>
                        """, unsafe_allow_html=True)
                except ValueError as e:
                    st.warning(f"⚠️ Backtest skipped: {e}")
        
        # ─── Feature Importance ───
        st.markdown('<div class="section-header">🧠 Model Insights</div>', unsafe_allow_html=True)
        col_fi, col_stats = st.columns([2, 1])
        
        with col_fi:
            st.plotly_chart(create_feature_importance_chart(result["feature_importance"]), use_container_width=True)
        
        with col_stats:
            st.markdown(f"""
            <div class="metric-card"><h3>Model Statistics</h3></div>
            
            | Metric | Value |
            |--------|-------|
            | **Algorithm** | Random Forest |
            | **Trees** | 200 |
            | **Training Samples** | {result['train_size']} |
            | **Test Samples** | {result['test_size']} |
            | **Test Accuracy** | {result['accuracy']:.1%} |
            | **Cross-Val Accuracy** | {result['cv_accuracy']:.1%} |
            | **Features Used** | {len(result['feature_names'])} |
            """)
        
        # ─── Key Indicator Values ───
        st.markdown('<div class="section-header">📊 Current Indicator Values</div>', unsafe_allow_html=True)
        ind_cols = st.columns(6)
        indicators_display = [
            ("RSI", f"{latest.get('RSI', 0):.1f}", "0-100"),
            ("MACD", f"{latest.get('MACD', 0):.4f}", "Momentum"),
            ("Volatility", f"{latest.get('Volatility', 0):.4f}", "10-day"),
            ("BB Position", f"{latest.get('BB_position', 0):.2f}", "0-1 range"),
            ("5D Return", f"{latest.get('Return_5d', 0):.2%}", "Weekly"),
            ("Vol Ratio", f"{latest.get('Volume_ratio', 0):.2f}", "vs 20d avg"),
        ]
        for i, (name, value, subtitle) in enumerate(indicators_display):
            with ind_cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3>{name}</h3>
                    <div class="value" style="font-size: 1.4rem;">{value}</div>
                    <div style="color: #6060a0; font-size: 0.8rem;">{subtitle}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # ─── Multi-Stock Comparison ───
        if compare_symbols.strip():
            st.markdown('<div class="section-header">📊 Multi-Stock Comparison</div>', unsafe_allow_html=True)
            compare_list = [s.strip().upper() for s in compare_symbols.split(",") if s.strip()]
            all_symbols = [symbol] + compare_list
            analyses = {}
            
            for sym in all_symbols:
                try:
                    with st.spinner(f"Analyzing {sym}..."):
                        sym_data = load_data(sym, period) if sym != symbol else data
                        if sym != symbol:
                            sym_data = add_all_indicators(sym_data)
                        sym_result = train_model(sym_data) if sym != symbol else result
                        sym_latest = sym_data.iloc[-1]
                        sym_decision = make_decision(
                            prediction=sym_result["prediction"], confidence=sym_result["confidence"],
                            rsi=sym_latest.get("RSI", 50), macd_hist=sym_latest.get("MACD_hist", 0),
                            volatility=sym_latest.get("Volatility", 0.02), bb_position=sym_latest.get("BB_position", 0.5)
                        )
                        sym_risk = calculate_risk(sym_data)
                        analyses[sym] = {
                            "decision": sym_decision, "risk": sym_risk,
                            "confidence": sym_result["confidence"], "accuracy": sym_result["accuracy"],
                            "price": sym_data["Close"].iloc[-1], "rsi": sym_latest.get("RSI", 50),
                        }
                except Exception as e:
                    st.warning(f"Could not analyze {sym}: {e}")
            
            if analyses:
                comp_data = []
                for sym, a in analyses.items():
                    comp_data.append({
                        "Symbol": sym, "Price": f"${a['price']:.2f}",
                        "Decision": f"{a['decision']['emoji']} {a['decision']['action']}",
                        "Confidence": f"{a['confidence']:.0%}", "RSI": f"{a['rsi']:.1f}",
                        "Risk": f"{a['risk']['level']} ({a['risk']['score']:.0f})",
                        "Accuracy": f"{a['accuracy']:.0%}",
                    })
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
                
                st.markdown('<div class="section-header">💼 Portfolio Allocation Suggestion</div>', unsafe_allow_html=True)
                portfolio = simple_portfolio_suggestion(analyses)
                for item in portfolio:
                    alloc_pct = item["allocation"] * 100
                    st.markdown(f"""
                    <div class="explanation-card">
                        <strong>{item['symbol']}</strong>: {alloc_pct:.1f}% allocation — {item['reason']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # NEW: PDF DOWNLOAD
        # ═══════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">📄 Download Report</div>', unsafe_allow_html=True)
        
        try:
            pdf_bytes = generate_report(
                symbol=symbol, stock_info=info, decision=decision, risk=risk,
                model_result=result, explanations=explanations,
                sentiment=sentiment_data, backtest=bt_result,
                advanced_metrics=advanced, support_resistance=sr_levels,
            )
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{symbol}_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"❌ PDF generation failed: {e}")
            import traceback
            st.expander("Show details").code(traceback.format_exc())
        
        # ─── Disclaimer ───
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Important Disclaimer:</strong> This analysis is generated by a machine learning model 
            for educational and informational purposes only. Past performance does not guarantee future results. 
            Stock markets are inherently unpredictable. The predictions and suggestions provided should NOT be 
            treated as financial advice. Always do your own research and consult with a qualified financial 
            advisor before making any investment decisions.
        </div>
        """, unsafe_allow_html=True)
        
    except ValueError as e:
        st.error(f"❌ Error: {e}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        st.exception(e)

else:
    # Landing state
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: #8080b0;">
        <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
        <h2 style="color: #a0a0ff; font-weight: 600;">Enter a Stock Symbol to Begin</h2>
        <p style="max-width: 600px; margin: 15px auto; line-height: 1.8; font-size: 1.05rem;">
            Use the sidebar to enter a stock ticker (e.g., <strong style="color:#c0c0ff;">AAPL</strong>, 
            <strong style="color:#c0c0ff;">GOOGL</strong>, <strong style="color:#c0c0ff;">TSLA</strong>) 
            and click <strong style="color:#6366f1;">Analyze Stock</strong>.
        </p>
        <div style="margin-top: 40px; display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size: 2rem;">🤖</div>
                <div style="color: #a0a0ff; font-weight: 600; margin-top: 8px;">ML Predictions</div>
                <div style="font-size: 0.85rem;">Random Forest</div>
            </div>
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size: 2rem;">💡</div>
                <div style="color: #a0a0ff; font-weight: 600; margin-top: 8px;">Explainable AI</div>
                <div style="font-size: 0.85rem;">Know WHY</div>
            </div>
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size: 2rem;">📉</div>
                <div style="color: #a0a0ff; font-weight: 600; margin-top: 8px;">Backtesting</div>
                <div style="font-size: 0.85rem;">Historical proof</div>
            </div>
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size: 2rem;">📰</div>
                <div style="color: #a0a0ff; font-weight: 600; margin-top: 8px;">Sentiment</div>
                <div style="font-size: 0.85rem;">News analysis</div>
            </div>
            <div style="text-align: center; min-width: 100px;">
                <div style="font-size: 2rem;">📄</div>
                <div style="color: #a0a0ff; font-weight: 600; margin-top: 8px;">PDF Reports</div>
                <div style="font-size: 0.85rem;">Download & share</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
