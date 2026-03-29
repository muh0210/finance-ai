# 🧠 AI Financial Decision Assistant

An ML-powered stock analysis tool that predicts market trends, scores risk, and **explains WHY** — so you don't just get a recommendation, you understand the reasoning.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## 📐 Architecture

```
User → Streamlit UI → Python Backend → ML Engine → Yahoo Finance API
```

| Component | File | Purpose |
|-----------|------|---------|
| Data Layer | `utils/data_loader.py` | Fetches stock data via yfinance with caching |
| Feature Engine | `utils/indicators.py` | 13 technical indicators (RSI, MACD, Bollinger, etc.) |
| ML Model | `utils/model.py` | Random Forest with time-series validation |
| Decision Engine | `utils/decision.py` | Buy/Sell/Hold + Risk Score + Explanations |
| UI | `app.py` | Streamlit dashboard with interactive charts |

## ✨ Features

- **🤖 ML Predictions** — Random Forest trained on 13 technical features
- **💡 Explainable AI** — Every decision comes with detailed reasoning
- **📉 Risk Scoring** — 4-factor risk assessment (Volatility, Drawdown, Volume, RSI)
- **📊 Portfolio Optimization** — Multi-stock comparison & allocation suggestions
- **📈 Interactive Charts** — Candlestick, RSI, MACD, Volume (Plotly)
- **⚡ Smart Caching** — API results cached locally for speed

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML (Random Forest Classifier)
- **yfinance** — Stock data API
- **Streamlit** — Web UI
- **Plotly** — Interactive charts
- **SHAP** — ML explainability (optional, for advanced use)

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does NOT constitute financial advice. Always consult a qualified financial advisor before making investment decisions.
