"""
report.py — PDF Report Generator

PURPOSE:
    Generate a downloadable PDF report of the stock analysis.
    Makes the tool feel like a REAL product — clients can save,
    print, or share the analysis.

WHAT'S IN THE REPORT:
    1. Stock header (name, symbol, price, date)
    2. AI Decision (Buy/Sell/Hold + confidence)
    3. Risk assessment
    4. Key indicators table
    5. Explanation of the decision
    6. News sentiment summary
    7. Disclaimer
"""

from fpdf import FPDF
from datetime import datetime
import io
import re


def sanitize(text: str) -> str:
    """Replace Unicode characters that Helvetica can't render with ASCII equivalents."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "*",   # bullet
        "\u00b7": "*",   # middle dot
        "\u2265": ">=",  # greater than or equal
        "\u2264": "<=",  # less than or equal
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining emoji/non-latin1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


class FinancialReport(FPDF):
    """Custom PDF class with financial report styling."""
    
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, sanitize("AI Financial Decision Assistant - Confidential Report"), align="R")
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")
    
    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 80)
        self.cell(0, 10, sanitize(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(80, 80, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
    
    def key_value(self, key, value, bold_value=False):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(60, 7, sanitize(f"{key}:"))
        self.set_font("Helvetica", "B" if bold_value else "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")


def generate_report(
    symbol: str,
    stock_info: dict,
    decision: dict,
    risk: dict,
    model_result: dict,
    explanations: list,
    sentiment: dict = None,
    backtest: dict = None,
    advanced_metrics: dict = None,
    support_resistance: dict = None,
) -> bytes:
    """
    Generate a complete PDF report.
    
    Returns bytes of the PDF file (can be downloaded via Streamlit).
    """
    pdf = FinancialReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # ─── TITLE ───
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 30, 80)
    pdf.cell(0, 12, "Stock Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    
    # ─── STOCK INFO ───
    pdf.section_title(f"{stock_info.get('name', symbol)} ({symbol})")
    pdf.key_value("Sector", stock_info.get("sector", "N/A"))
    pdf.key_value("Industry", stock_info.get("industry", "N/A"))
    
    # Format market cap
    cap = stock_info.get("market_cap", 0)
    if cap >= 1e12:
        cap_str = f"${cap/1e12:.2f}T"
    elif cap >= 1e9:
        cap_str = f"${cap/1e9:.2f}B"
    elif cap >= 1e6:
        cap_str = f"${cap/1e6:.2f}M"
    else:
        cap_str = f"${cap:,.0f}"
    pdf.key_value("Market Cap", cap_str)
    pdf.key_value("52-Week High", f"${stock_info.get('52w_high', 0):.2f}")
    pdf.key_value("52-Week Low", f"${stock_info.get('52w_low', 0):.2f}")
    pdf.ln(5)
    
    # ─── AI DECISION ───
    pdf.section_title("AI Decision")
    
    action = decision.get("action", "HOLD")
    pdf.set_font("Helvetica", "B", 18)
    if "BUY" in action:
        pdf.set_text_color(0, 150, 0)
    elif "SELL" in action:
        pdf.set_text_color(200, 0, 0)
    else:
        pdf.set_text_color(200, 150, 0)
    pdf.cell(0, 12, sanitize(f"{decision.get('emoji', '')} {action}"), new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_text_color(30, 30, 30)
    pdf.key_value("Model Confidence", f"{model_result.get('confidence', 0):.0%}", bold_value=True)
    pdf.key_value("Model Accuracy (Test)", f"{model_result.get('accuracy', 0):.0%}")
    pdf.key_value("Cross-Validation Accuracy", f"{model_result.get('cv_accuracy', 0):.0%}")
    pdf.ln(5)
    
    # ─── RISK ASSESSMENT ───
    pdf.section_title("Risk Assessment")
    
    risk_level = risk.get("level", "N/A")
    risk_score = risk.get("score", 0)
    pdf.set_font("Helvetica", "B", 14)
    if risk_level in ["High", "Very High"]:
        pdf.set_text_color(200, 0, 0)
    elif risk_level == "Medium":
        pdf.set_text_color(200, 150, 0)
    else:
        pdf.set_text_color(0, 150, 0)
    pdf.cell(0, 10, f"{risk_level} Risk ({risk_score:.0f}/100)", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_text_color(30, 30, 30)
    for factor_text, emoji, level in risk.get("factors", []):
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, sanitize(f"  {emoji} {factor_text}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # ─── EXPLANATION ───
    pdf.section_title("Why This Decision?")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    
    for exp in explanations:
        # Strip markdown bold markers for PDF
        clean = exp.replace("**", "").replace("*", "")
        pdf.multi_cell(0, 6, sanitize(clean))
        pdf.ln(3)
    
    # ─── ADVANCED METRICS ───
    if advanced_metrics:
        pdf.add_page()
        pdf.section_title("Advanced Financial Metrics")
        pdf.key_value("Annualized Return", f"{advanced_metrics['annualized_return']:.2%}", bold_value=True)
        pdf.key_value("Annualized Volatility", f"{advanced_metrics['annualized_volatility']:.2%}")
        pdf.key_value("Sharpe Ratio", f"{advanced_metrics['sharpe_ratio']:.2f}", bold_value=True)
        pdf.key_value("Sortino Ratio", f"{advanced_metrics['sortino_ratio']:.2f}")
        pdf.key_value("Calmar Ratio", f"{advanced_metrics['calmar_ratio']:.2f}")
        pdf.key_value("Max Drawdown", f"{advanced_metrics['max_drawdown']:.2%}")
        pdf.key_value("Value at Risk (95%)", f"{advanced_metrics['var_95']:.2%}")
        pdf.key_value("Profit Factor", f"{advanced_metrics['profit_factor']:.2f}")
        pdf.key_value("Best Day", f"{advanced_metrics['best_day']:.2%}")
        pdf.key_value("Worst Day", f"{advanced_metrics['worst_day']:.2%}")
        pdf.key_value("Positive Days", str(advanced_metrics['positive_days']))
        pdf.key_value("Negative Days", str(advanced_metrics['negative_days']))
        pdf.ln(5)
    
    # ─── SUPPORT & RESISTANCE ───
    if support_resistance:
        pdf.section_title("Support & Resistance Levels")
        pdf.key_value("Current Price", f"${support_resistance['current_price']:.2f}", bold_value=True)
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Resistance Levels (ceiling):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for r in support_resistance.get("resistance", []):
            pdf.cell(0, 6, f"  ${r['price']:.2f} (strength: {r['strength']} touches)", new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Support Levels (floor):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for s in support_resistance.get("support", []):
            pdf.cell(0, 6, f"  ${s['price']:.2f} (strength: {s['strength']} touches)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
    
    # ─── BACKTEST RESULTS ───
    if backtest:
        pdf.section_title("Backtest Results")
        pdf.key_value("Strategy Return", f"{backtest['total_return_strategy']:.2f}%", bold_value=True)
        pdf.key_value("Buy & Hold Return", f"{backtest['total_return_buyhold']:.2f}%")
        pdf.key_value("Backtest Accuracy", f"{backtest['accuracy']:.0%}")
        pdf.key_value("Sharpe Ratio", f"{backtest['sharpe_ratio']:.2f}")
        pdf.key_value("Max Drawdown", f"{backtest['max_drawdown']:.2f}%")
        pdf.key_value("Win Rate", f"{backtest['win_rate']:.0%}")
        pdf.key_value("Total Trades", str(backtest['trades']))
        pdf.key_value("Days Tested", str(backtest['days_tested']))
        pdf.ln(5)
    
    # ─── NEWS SENTIMENT ───
    if sentiment and sentiment.get("articles"):
        pdf.section_title("News Sentiment Analysis")
        pdf.key_value("Overall Sentiment", f"{sentiment['overall_label']} ({sentiment['overall_score']:+.2f})", bold_value=True)
        pdf.key_value("Positive Articles", str(sentiment['positive_count']))
        pdf.key_value("Negative Articles", str(sentiment['negative_count']))
        pdf.key_value("Neutral Articles", str(sentiment['neutral_count']))
        pdf.ln(3)
        
        pdf.set_font("Helvetica", "", 9)
        for article in sentiment['articles'][:5]:  # Top 5
            pdf.set_text_color(50, 50, 50)
            headline = article['headline'][:80] + ("..." if len(article['headline']) > 80 else "")
            pdf.cell(0, 6, sanitize(f"  {article['emoji']} {headline}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
    
    # ─── DISCLAIMER ───
    pdf.add_page()
    pdf.section_title("Disclaimer")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    disclaimer = (
        "This report is generated by an AI-powered machine learning model for educational and informational "
        "purposes only. It does NOT constitute financial advice, investment recommendations, or any solicitation "
        "to buy or sell securities. Past performance does not guarantee future results. Stock markets are "
        "inherently unpredictable and involve substantial risk of loss. The predictions, suggestions, and "
        "analysis provided should NOT be treated as the sole basis for investment decisions. Always conduct "
        "your own research and consult with a qualified financial advisor before making any investment decisions. "
        "The creators and operators of this tool assume no liability for any financial losses incurred as a "
        "result of using this tool or acting on its output."
    )
    pdf.multi_cell(0, 5, disclaimer)
    
    # Return as bytes
    return pdf.output()
