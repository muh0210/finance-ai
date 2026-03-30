# 🧠 AI Financial Decision Assistant

> **ML-Powered Stock Analysis** · Backtesting · News Sentiment · PDF Reports

An intelligent financial analysis platform that combines **Machine Learning predictions** with **real-time news sentiment**, **historical backtesting**, and **advanced financial metrics** — then explains every decision in plain English with a downloadable PDF report.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finance-ai.streamlit.app/)

---

## 🎯 What It Does

Enter any stock ticker → Get a **BUY / SELL / HOLD** recommendation backed by:

| Feature | Description |
|---------|-------------|
| 🤖 **ML Predictions** | Random Forest trained on 13 technical indicators with time-series cross-validation |
| 💡 **Explainable AI** | Every decision comes with detailed, human-readable reasoning |
| 📉 **Backtesting** | Walk-forward historical simulation — proves if the model would have made money |
| 📰 **News Sentiment** | Real-time headline analysis from Google News with financial keyword scoring |
| 📏 **Support & Resistance** | Automatic detection of price floors and ceilings |
| ⭐ **Advanced Metrics** | Sharpe, Sortino, Calmar ratios, VaR, Max Drawdown, Profit Factor |
| 📄 **PDF Reports** | Professional, downloadable reports for clients or records |
| 📊 **Multi-Stock Comparison** | Compare multiple stocks side-by-side with portfolio allocation suggestions |
| 🎨 **Premium UI** | Dark-themed, glassmorphism-styled dashboard with interactive Plotly charts |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/muh0210/finance-ai.git
cd finance-ai

# 2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📐 Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Streamlit   │────▶│  Data Pipeline  │────▶│  Yahoo Finance   │
│  Dashboard   │     │  (data_loader)  │     │  API + Cache     │
└──────┬───────┘     └────────┬────────┘     └──────────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Decision   │◀────│  ML Engine      │◀────│  Feature Engine  │
│   Engine     │     │  (model.py)     │     │  (indicators.py) │
└──────┬───────┘     └─────────────────┘     └──────────────────┘
       │
       ├──▶ Risk Assessment
       ├──▶ Explanation Generation
       ├──▶ News Sentiment Analysis
       ├──▶ Backtesting Engine
       └──▶ PDF Report Generator
```

### Module Breakdown

| Module | File | Purpose |
|--------|------|---------|
| **Data Layer** | `utils/data_loader.py` | Fetches & caches stock data via yfinance, handles MultiIndex columns |
| **Feature Engine** | `utils/indicators.py` | 13 technical indicators: RSI, MACD, Bollinger Bands, Moving Averages, Volatility, Volume features, Price position |
| **ML Model** | `utils/model.py` | Random Forest Classifier (200 trees) with time-series cross-validation |
| **Decision Engine** | `utils/decision.py` | Multi-factor BUY/SELL/HOLD + 4-factor Risk Score + Human-readable explanations |
| **Backtesting** | `utils/backtest.py` | Walk-forward simulation + Support/Resistance detection + Advanced financial metrics |
| **Sentiment** | `utils/sentiment.py` | Google News RSS + Financial keyword sentiment scoring |
| **PDF Reports** | `utils/report.py` | Professional PDF generation with safe Unicode handling |
| **UI** | `app.py` | Streamlit dashboard with dark theme, interactive charts, download buttons |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | Scikit-learn (Random Forest, Gradient Boosting) |
| **Data** | Pandas, NumPy |
| **Stock API** | yfinance (Yahoo Finance) |
| **Web Framework** | Streamlit |
| **Charts** | Plotly (Candlestick, RSI, MACD, Volume, Equity Curves) |
| **Sentiment** | feedparser + Custom Financial Lexicon |
| **PDF** | fpdf2 |
| **Explainability** | SHAP (optional) |

---

## 📊 How the ML Model Works

1. **Feature Engineering** — 13 technical indicators computed from raw OHLCV data
2. **Target** — Binary classification: Will tomorrow's close be higher than today's?
3. **Training** — Time-series split (80/20) respecting temporal order — no data leakage
4. **Validation** — 5-fold TimeSeriesSplit cross-validation for robustness
5. **Ensemble** — 200-tree Random Forest with balanced class weights
6. **Decision** — ML output combined with RSI, MACD, Bollinger position, and volatility for composite scoring

---

## 📁 Project Structure

```
finance-ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
├── models/
│   └── latest_model.pkl    # Saved trained model (auto-generated)
├── data/
│   └── *.csv               # Cached stock data (auto-generated)
└── utils/
    ├── __init__.py
    ├── data_loader.py       # Stock data fetching & caching
    ├── indicators.py        # Technical indicator calculations
    ├── model.py             # ML model training & prediction
    ├── decision.py          # Decision engine + risk + explanations
    ├── backtest.py          # Historical backtesting + advanced metrics
    ├── sentiment.py         # News sentiment analysis
    └── report.py            # PDF report generation
```

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does NOT constitute financial advice, investment recommendations, or any solicitation to buy or sell securities. Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.

---

## 📜 License

MIT License — free to use, modify, and distribute.
