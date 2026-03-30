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
import math


# ═══════════════════════════════════════════════════════════════════════════
# SANITIZATION — Strip everything Helvetica/Latin-1 can't render
# ═══════════════════════════════════════════════════════════════════════════

# Regex that matches emoji and miscellaneous symbol blocks
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U00002640-\U00002642"  # gender symbols
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000203C-\U00003299"  # CJK symbols / enclosed
    "]+",
    flags=re.UNICODE,
)


def sanitize(text: str) -> str:
    """Replace Unicode characters that Helvetica can't render with ASCII equivalents."""
    if text is None:
        return ""
    text = str(text)

    # Named unicode replacements
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
        "\u2248": "~",   # approximately equal
        "\u00a0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Strip ALL emoji/symbol characters
    text = _EMOJI_RE.sub("", text)

    # Final pass: encode to latin-1, replacing anything left over
    text = text.encode("latin-1", errors="replace").decode("latin-1")

    # Clean up multiple spaces left by stripped emoji
    text = re.sub(r"  +", " ", text).strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════
# SAFE FORMATTING — Handle None / NaN / missing values
# ═══════════════════════════════════════════════════════════════════════════

def safe_float(value, default=0.0):
    """Convert value to float safely, returning default for None/NaN/errors."""
    if value is None:
        return default
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_fmt(value, fmt=".2f", default="N/A", prefix="", suffix=""):
    """Safely format a numeric value, returning default string on failure."""
    val = safe_float(value, default=None)
    if val is None:
        return str(default)
    try:
        return f"{prefix}{val:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(default)


def safe_pct(value, fmt=".0%", default="N/A"):
    """Safely format a value as percentage."""
    val = safe_float(value, default=None)
    if val is None:
        return str(default)
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(default)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM PDF CLASS
# ═══════════════════════════════════════════════════════════════════════════

# Minimum Y position before we force a new page (in mm from top)
PAGE_BOTTOM_MARGIN = 260


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

    def check_page_break(self, needed_height=40):
        """Add a page break if remaining space is insufficient."""
        if self.get_y() + needed_height > PAGE_BOTTOM_MARGIN:
            self.add_page()

    def section_title(self, title):
        self.check_page_break(30)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 80)
        self.cell(0, 10, sanitize(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(80, 80, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def key_value(self, key, value, bold_value=False):
        self.check_page_break(10)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(60, 7, sanitize(f"{key}:"))
        self.set_font("Helvetica", "B" if bold_value else "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")

    def safe_multi_cell(self, w, h, txt):
        """Write multi_cell with page break check and sanitization."""
        self.check_page_break(20)
        self.multi_cell(w, h, sanitize(txt))


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

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
    All inputs are accessed defensively — missing keys won't crash.
    """
    # Ensure inputs are dicts (protect against None being passed for required params)
    stock_info = stock_info or {}
    decision = decision or {}
    risk = risk or {}
    model_result = model_result or {}
    explanations = explanations or []

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
    stock_name = sanitize(stock_info.get("name", symbol) or symbol)
    pdf.section_title(f"{stock_name} ({symbol})")
    pdf.key_value("Sector", stock_info.get("sector", "N/A") or "N/A")
    pdf.key_value("Industry", stock_info.get("industry", "N/A") or "N/A")

    # Format market cap
    cap = safe_float(stock_info.get("market_cap", 0))
    if cap >= 1e12:
        cap_str = f"${cap/1e12:.2f}T"
    elif cap >= 1e9:
        cap_str = f"${cap/1e9:.2f}B"
    elif cap >= 1e6:
        cap_str = f"${cap/1e6:.2f}M"
    else:
        cap_str = f"${cap:,.0f}"
    pdf.key_value("Market Cap", cap_str)
    pdf.key_value("52-Week High", safe_fmt(stock_info.get("52w_high"), prefix="$"))
    pdf.key_value("52-Week Low", safe_fmt(stock_info.get("52w_low"), prefix="$"))
    pdf.ln(5)

    # ─── AI DECISION ───
    pdf.section_title("AI Decision")

    action = sanitize(decision.get("action", "HOLD"))
    pdf.set_font("Helvetica", "B", 18)
    if "BUY" in action.upper():
        pdf.set_text_color(0, 150, 0)
    elif "SELL" in action.upper():
        pdf.set_text_color(200, 0, 0)
    else:
        pdf.set_text_color(200, 150, 0)
    pdf.cell(0, 12, sanitize(action), new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(30, 30, 30)
    pdf.key_value("Model Confidence", safe_pct(model_result.get("confidence")), bold_value=True)
    pdf.key_value("Model Accuracy (Test)", safe_pct(model_result.get("accuracy")))
    pdf.key_value("Cross-Validation Accuracy", safe_pct(model_result.get("cv_accuracy")))
    pdf.ln(5)

    # ─── RISK ASSESSMENT ───
    pdf.section_title("Risk Assessment")

    risk_level = risk.get("level", "N/A") or "N/A"
    risk_score = safe_float(risk.get("score", 0))
    pdf.set_font("Helvetica", "B", 14)
    if risk_level in ["High", "Very High"]:
        pdf.set_text_color(200, 0, 0)
    elif risk_level == "Medium":
        pdf.set_text_color(200, 150, 0)
    else:
        pdf.set_text_color(0, 150, 0)
    pdf.cell(0, 10, f"{risk_level} Risk ({risk_score:.0f}/100)", new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(30, 30, 30)
    for factor in risk.get("factors", []):
        # factors is a list of tuples: (text, emoji, level)
        if isinstance(factor, (list, tuple)) and len(factor) >= 1:
            factor_text = sanitize(str(factor[0]))
        else:
            factor_text = sanitize(str(factor))
        pdf.check_page_break(10)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, sanitize(f"  - {factor_text}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ─── EXPLANATION ───
    pdf.section_title("Why This Decision?")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)

    for exp in explanations:
        # Strip markdown bold markers for PDF
        clean = str(exp).replace("**", "").replace("*", "")
        pdf.check_page_break(20)
        pdf.safe_multi_cell(0, 6, clean)
        pdf.ln(3)

    # ─── ADVANCED METRICS ───
    if advanced_metrics:
        pdf.add_page()
        pdf.section_title("Advanced Financial Metrics")
        pdf.key_value("Annualized Return", safe_pct(advanced_metrics.get("annualized_return"), fmt=".2%"), bold_value=True)
        pdf.key_value("Annualized Volatility", safe_pct(advanced_metrics.get("annualized_volatility"), fmt=".2%"))
        pdf.key_value("Sharpe Ratio", safe_fmt(advanced_metrics.get("sharpe_ratio")), bold_value=True)
        pdf.key_value("Sortino Ratio", safe_fmt(advanced_metrics.get("sortino_ratio")))
        pdf.key_value("Calmar Ratio", safe_fmt(advanced_metrics.get("calmar_ratio")))
        pdf.key_value("Max Drawdown", safe_pct(advanced_metrics.get("max_drawdown"), fmt=".2%"))
        pdf.key_value("Value at Risk (95%)", safe_pct(advanced_metrics.get("var_95"), fmt=".2%"))
        pdf.key_value("Profit Factor", safe_fmt(advanced_metrics.get("profit_factor")))
        pdf.key_value("Best Day", safe_pct(advanced_metrics.get("best_day"), fmt=".2%"))
        pdf.key_value("Worst Day", safe_pct(advanced_metrics.get("worst_day"), fmt=".2%"))
        pdf.key_value("Positive Days", str(advanced_metrics.get("positive_days", "N/A")))
        pdf.key_value("Negative Days", str(advanced_metrics.get("negative_days", "N/A")))
        pdf.ln(5)

    # ─── SUPPORT & RESISTANCE ───
    if support_resistance:
        pdf.section_title("Support & Resistance Levels")
        pdf.key_value("Current Price", safe_fmt(support_resistance.get("current_price"), prefix="$"), bold_value=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Resistance Levels (ceiling):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        resistance_levels = support_resistance.get("resistance", [])
        if resistance_levels:
            for r in resistance_levels:
                price = safe_fmt(r.get("price") if isinstance(r, dict) else r, prefix="$")
                strength = r.get("strength", "?") if isinstance(r, dict) else "?"
                pdf.check_page_break(10)
                pdf.cell(0, 6, f"  {price} (strength: {strength} touches)", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 6, "  No resistance levels detected", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Support Levels (floor):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        support_levels = support_resistance.get("support", [])
        if support_levels:
            for s in support_levels:
                price = safe_fmt(s.get("price") if isinstance(s, dict) else s, prefix="$")
                strength = s.get("strength", "?") if isinstance(s, dict) else "?"
                pdf.check_page_break(10)
                pdf.cell(0, 6, f"  {price} (strength: {strength} touches)", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 6, "  No support levels detected", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # ─── BACKTEST RESULTS ───
    if backtest:
        pdf.section_title("Backtest Results")
        pdf.key_value("Strategy Return", safe_fmt(backtest.get("total_return_strategy"), suffix="%"), bold_value=True)
        pdf.key_value("Buy & Hold Return", safe_fmt(backtest.get("total_return_buyhold"), suffix="%"))
        pdf.key_value("Backtest Accuracy", safe_pct(backtest.get("accuracy")))
        pdf.key_value("Sharpe Ratio", safe_fmt(backtest.get("sharpe_ratio")))
        pdf.key_value("Max Drawdown", safe_fmt(backtest.get("max_drawdown"), suffix="%"))
        pdf.key_value("Win Rate", safe_pct(backtest.get("win_rate")))
        pdf.key_value("Total Trades", str(backtest.get("trades", "N/A")))
        pdf.key_value("Days Tested", str(backtest.get("days_tested", "N/A")))
        pdf.ln(5)

    # ─── NEWS SENTIMENT ───
    if sentiment and sentiment.get("articles"):
        pdf.section_title("News Sentiment Analysis")
        overall_label = sentiment.get("overall_label", "N/A")
        overall_score = safe_float(sentiment.get("overall_score", 0))
        pdf.key_value("Overall Sentiment", f"{overall_label} ({overall_score:+.2f})", bold_value=True)
        pdf.key_value("Positive Articles", str(sentiment.get("positive_count", 0)))
        pdf.key_value("Negative Articles", str(sentiment.get("negative_count", 0)))
        pdf.key_value("Neutral Articles", str(sentiment.get("neutral_count", 0)))
        pdf.ln(3)

        pdf.set_font("Helvetica", "", 9)
        for article in sentiment.get("articles", [])[:5]:  # Top 5
            pdf.set_text_color(50, 50, 50)
            headline = sanitize(str(article.get("headline", "No headline")))
            if len(headline) > 80:
                headline = headline[:80] + "..."
            pdf.check_page_break(10)
            pdf.cell(0, 6, sanitize(f"  - {headline}"), new_x="LMARGIN", new_y="NEXT")
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
    return bytes(pdf.output())
