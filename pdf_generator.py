"""
pdf_generator.py
Generates a downloadable PDF research report summarizing stock metrics,
technical analysis, news, and the final recommendation.
"""

import os
import re
import unicodedata
from datetime import datetime
from fpdf import FPDF

from utils import format_large_number, format_percentage

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Core FPDF fonts (Helvetica, Times, Courier) only support latin-1.
# Any character outside that range (curly quotes, em-dashes, emoji, etc.
# sometimes returned by the LLM or news feeds) will otherwise crash the PDF.
_LATIN1_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "-",
    "\u00a0": " ",
}


def _sanitize_text(text: str, max_word_len: int = 45) -> str:
    """
    Make arbitrary text safe to render with FPDF's core fonts:
      1. Normalize/replace common non-latin-1 punctuation.
      2. Strip any remaining character outside latin-1.
      3. Break up any single "word" (no spaces) longer than max_word_len
         by inserting a space every max_word_len characters. This is the
         key fix for "Not enough horizontal space to render a single
         character" - that error happens when FPDF hits an unbroken
         string (long URL, ticker spam, run-on token from an LLM, etc.)
         that is wider than the available cell width and has nowhere to
         wrap.
    """
    if text is None:
        return ""
    text = str(text)

    for bad, good in _LATIN1_REPLACEMENTS.items():
        text = text.replace(bad, good)

    # Drop any character that still can't be encoded in latin-1
    text = text.encode("latin-1", "ignore").decode("latin-1")

    def _break_long_word(match):
        word = match.group(0)
        return " ".join(word[i:i + max_word_len] for i in range(0, len(word), max_word_len))

    text = re.sub(r"\S{%d,}" % (max_word_len + 1), _break_long_word, text)

    return text


class PDFReport(FPDF):
    """Custom FPDF subclass with a consistent header/footer for the report."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "Stock Research Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_draw_color(200, 200, 200)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(20, 60, 120)
        self.cell(0, 8, _sanitize_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def key_value_row(self, key: str, value: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.cell(60, 7, _sanitize_text(key), new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "", 10)
        # Available width = page width - current x - right margin
        available_width = self.w - self.r_margin - self.get_x()
        if available_width < 10:
            # Safety net: if somehow there isn't enough room, drop to a new line
            self.ln(7)
            self.set_x(self.l_margin)
            available_width = self.w - self.r_margin - self.get_x()
        self.multi_cell(available_width, 7, _sanitize_text(value))

    def body_text(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 6, _sanitize_text(text))
        self.ln(1)

    def bullet_line(self, text: str):
        self.set_x(self.l_margin)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 6, _sanitize_text(text))


def _safe(value, default="N/A"):
    if value is None:
        return default
    return str(value)


def generate_pdf_report(research_data: dict, recommendation_data: dict) -> str:
    """
    Build a PDF report from research_data (output of ResearchAgent) and
    recommendation_data (output of RecommendationAgent). Saves the file
    into the reports/ directory and returns the full file path.
    """
    stock_info = research_data["stock_info"]
    technicals = research_data["technicals"]
    sentiment = research_data["sentiment"]
    news = research_data.get("news", [])
    ticker = research_data["ticker"]

    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # --- Title block ---
    pdf.set_font("Helvetica", "B", 16)
    title_line = f"{ticker} - {stock_info.get('company_name', 'N/A')}"
    pdf.multi_cell(0, 10, _sanitize_text(title_line))
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # --- Key Metrics ---
    pdf.section_title("Key Metrics")
    pdf.key_value_row("Current Price:", f"${_safe(stock_info.get('current_price'))}")
    pdf.key_value_row("Market Cap:", format_large_number(stock_info.get("market_cap")))
    pdf.key_value_row("PE Ratio:", _safe(stock_info.get("pe_ratio")))
    pdf.key_value_row("Dividend Yield:", format_percentage(stock_info.get("dividend_yield")))
    pdf.key_value_row("52 Week High:", f"${_safe(stock_info.get('fifty_two_week_high'))}")
    pdf.key_value_row("52 Week Low:", f"${_safe(stock_info.get('fifty_two_week_low'))}")
    pdf.key_value_row("Sector:", _safe(stock_info.get("sector")))
    pdf.key_value_row("Industry:", _safe(stock_info.get("industry")))
    pdf.ln(3)

    # --- Technical Analysis ---
    pdf.section_title("Technical Analysis")
    pdf.key_value_row("RSI (14):", _safe(technicals.get("rsi")))
    pdf.key_value_row("50 Day MA:", _safe(technicals.get("ma50")))
    pdf.key_value_row("200 Day MA:", _safe(technicals.get("ma200")))
    pdf.key_value_row("Volatility (Annualized):", f"{_safe(technicals.get('volatility'))}%")
    pdf.ln(3)

    # --- News & Sentiment Summary ---
    pdf.section_title("News & Sentiment Summary")
    pdf.key_value_row("Positive Articles:", _safe(sentiment.get("positive_count")))
    pdf.key_value_row("Negative Articles:", _safe(sentiment.get("negative_count")))
    pdf.key_value_row("Neutral Articles:", _safe(sentiment.get("neutral_count")))
    pdf.key_value_row("Sentiment Score (0-100):", _safe(sentiment.get("sentiment_score")))
    pdf.ln(2)

    if news:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Recent Headlines:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for article in news[:8]:
            title = (article.get("title") or "").strip()
            source = article.get("source", "Unknown")
            if title:
                pdf.bullet_line(f"- {title} ({source})")
        pdf.ln(2)
    else:
        pdf.body_text("No recent news articles were available.")

    # --- Recommendation ---
    pdf.section_title("Recommendation")
    pdf.set_font("Helvetica", "B", 13)
    rec = recommendation_data.get("recommendation", "HOLD")
    color_map = {"BUY": (0, 130, 0), "SELL": (200, 0, 0), "HOLD": (200, 140, 0)}
    r, g, b = color_map.get(rec, (0, 0, 0))
    pdf.set_text_color(r, g, b)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 10, rec, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.key_value_row("Confidence Score:", f"{_safe(recommendation_data.get('confidence_score'))}/100")
    pdf.key_value_row("Market Sentiment:", _safe(recommendation_data.get("market_sentiment")))
    pdf.key_value_row("Rule-Based Signal:", _safe(recommendation_data.get("rule_based_signal")))
    pdf.ln(2)

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Rationale:", new_x="LMARGIN", new_y="NEXT")
    pdf.body_text(_safe(recommendation_data.get("rationale")))

    # --- Save file ---
    filename = f"{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)
    pdf.output(filepath)

    return filepath