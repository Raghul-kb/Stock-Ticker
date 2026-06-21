"""
utils.py
Utility functions for fetching stock data, computing technical indicators,
fetching news, and calculating a simple sentiment score.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# ---------------------------------------------------------------------------
# Yahoo Finance frequently blocks/rate-limits requests that look like a
# generic Python script. Using curl_cffi to impersonate a real browser's
# TLS fingerprint is the most reliable current workaround. We build one
# session and reuse it across calls. If curl_cffi isn't installed, we
# silently fall back to yfinance's default behavior.
# ---------------------------------------------------------------------------
_YF_SESSION = None


def _get_yf_session():
    global _YF_SESSION
    if _YF_SESSION is not None:
        return _YF_SESSION
    try:
        from curl_cffi import requests as cffi_requests
        _YF_SESSION = cffi_requests.Session(impersonate="chrome")
    except Exception:
        _YF_SESSION = None
    return _YF_SESSION


# ---------------------------------------------------------------------------
# Simple keyword lists used for naive sentiment scoring of news headlines.
# This is intentionally lightweight (no heavy NLP dependency) and is meant
# as a quick heuristic signal, not a precise sentiment classifier.
# ---------------------------------------------------------------------------
POSITIVE_WORDS = [
    "beat", "beats", "surge", "surges", "rally", "rallies", "growth", "gain",
    "gains", "profit", "profits", "upgrade", "upgraded", "bullish", "strong",
    "record", "outperform", "rise", "rises", "rising", "soar", "soars",
    "positive", "buy", "boost", "boosts", "exceed", "exceeds", "optimism",
    "win", "wins", "success", "successful", "expand", "expansion"
]

NEGATIVE_WORDS = [
    "miss", "misses", "plunge", "plunges", "fall", "falls", "falling",
    "drop", "drops", "loss", "losses", "downgrade", "downgraded", "bearish",
    "weak", "decline", "declines", "underperform", "slump", "slumps",
    "negative", "sell", "cut", "cuts", "lawsuit", "investigation", "fraud",
    "recall", "layoff", "layoffs", "crash", "crashes", "concern", "concerns",
    "risk", "risks", "warning", "warns"
]


class DataFetchError(Exception):
    """Custom exception raised when stock or news data cannot be retrieved."""
    pass


def fetch_stock_info(ticker: str, max_retries: int = 3, retry_delay: float = 2.0) -> dict:
    """
    Fetch core stock information using yFinance.

    Yahoo Finance's endpoints frequently rate-limit or block requests that
    look like a plain Python script, and/or occasionally return an empty
    or non-JSON body. To handle this robustly we:
      1. Use a curl_cffi session that impersonates a real browser when
         available (falls back to default yfinance behavior otherwise).
      2. Retry a few times with a short delay on transient failures.
      3. Fetch price history first (most reliable call), then try the
         richer `.info` metadata endpoint and fall back to `.fast_info` /
         history-derived values if `.info` fails.
    """
    ticker = ticker.strip().upper()
    session = _get_yf_session()

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            stock = yf.Ticker(ticker, session=session) if session else yf.Ticker(ticker)

            history = stock.history(period="1y")

            if history is None or history.empty:
                last_error = DataFetchError(
                    f"No data found for ticker '{ticker}'. Please check the symbol is correct "
                    f"(e.g. 'AAPL', 'MSFT')."
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                raise last_error

            # --- try the rich metadata endpoint (.info), but don't fail hard ---
            info = {}
            try:
                info = stock.info or {}
            except Exception:
                info = {}

            # --- fallback to fast_info if .info came back empty/broken ---
            fast_info = {}
            if not info:
                try:
                    fast_info = dict(stock.fast_info) if stock.fast_info else {}
                except Exception:
                    fast_info = {}

            def _get(*keys, default=None):
                for source in (info, fast_info):
                    for key in keys:
                        if source and key in source and source[key] is not None:
                            return source[key]
                return default

            current_price = _get("currentPrice", "regularMarketPrice", "lastPrice", "last_price")
            if current_price is None:
                current_price = float(history["Close"].iloc[-1])

            fifty_two_week_high = _get("fiftyTwoWeekHigh", "year_high")
            if fifty_two_week_high is None:
                fifty_two_week_high = round(float(history["High"].max()), 2)

            fifty_two_week_low = _get("fiftyTwoWeekLow", "year_low")
            if fifty_two_week_low is None:
                fifty_two_week_low = round(float(history["Low"].min()), 2)

            data = {
                "ticker": ticker,
                "company_name": _get("longName", "shortName", default=ticker),
                "current_price": round(float(current_price), 2) if current_price is not None else None,
                "market_cap": _get("marketCap", "market_cap"),
                "pe_ratio": _get("trailingPE"),
                "dividend_yield": _get("dividendYield"),
                "fifty_two_week_high": fifty_two_week_high,
                "fifty_two_week_low": fifty_two_week_low,
                "sector": _get("sector", default="N/A"),
                "industry": _get("industry", default="N/A"),
                "history": history
            }
            return data

        except DataFetchError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise

        except Exception as e:
            last_error = DataFetchError(f"Failed to fetch stock info for '{ticker}': {str(e)}")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise last_error

    # Should not normally reach here, but guard just in case
    raise last_error or DataFetchError(f"Failed to fetch stock info for '{ticker}'.")


def calculate_technical_indicators(history: pd.DataFrame) -> dict:
    """
    Calculate RSI, 50-day MA, 200-day MA, and annualized volatility
    from a price history dataframe (must contain a 'Close' column).
    """
    try:
        if history is None or history.empty:
            raise DataFetchError("No price history available to calculate technical indicators.")

        close = history["Close"].dropna()

        # RSI (14-period default)
        rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        rsi = round(float(rsi_series.iloc[-1]), 2) if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]) else None

        # Moving averages
        ma50 = round(float(close.rolling(window=50).mean().iloc[-1]), 2) if len(close) >= 50 else None
        ma200 = round(float(close.rolling(window=200).mean().iloc[-1]), 2) if len(close) >= 200 else None

        # Annualized volatility based on daily returns
        daily_returns = close.pct_change().dropna()
        volatility = round(float(daily_returns.std() * np.sqrt(252) * 100), 2) if not daily_returns.empty else None

        return {
            "rsi": rsi,
            "ma50": ma50,
            "ma200": ma200,
            "volatility": volatility
        }
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"Failed to calculate technical indicators: {str(e)}")


def fetch_news(ticker: str, company_name: str = "", page_size: int = 15) -> list:
    """
    Fetch latest news articles related to the ticker/company using NewsAPI.
    Returns a list of dicts with 'title', 'description', 'source', 'url', 'publishedAt'.
    """
    if not NEWS_API_KEY:
        raise DataFetchError("NEWS_API_KEY not found. Please set it in your .env file.")

    query = company_name if company_name else ticker

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "ok":
            raise DataFetchError(f"NewsAPI error: {payload.get('message', 'Unknown error')}")

        articles = payload.get("articles", [])
        news_list = []
        for article in articles:
            news_list.append({
                "title": article.get("title") or "",
                "description": article.get("description") or "",
                "source": (article.get("source") or {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", "")
            })
        return news_list

    except requests.exceptions.RequestException as e:
        raise DataFetchError(f"Failed to fetch news: {str(e)}")
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"Unexpected error fetching news: {str(e)}")


def calculate_sentiment(news_list: list) -> dict:
    """
    Calculate a simple keyword-based sentiment score from news headlines
    and descriptions.

    Returns positive/negative/neutral counts plus a 0-100 sentiment score
    where 50 is neutral, >50 leans positive, <50 leans negative.
    """
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    if not news_list:
        return {
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "sentiment_score": 50.0
        }

    for article in news_list:
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()

        pos_hits = sum(1 for word in POSITIVE_WORDS if word in text)
        neg_hits = sum(1 for word in NEGATIVE_WORDS if word in text)

        if pos_hits > neg_hits:
            positive_count += 1
        elif neg_hits > pos_hits:
            negative_count += 1
        else:
            neutral_count += 1

    total = positive_count + negative_count + neutral_count
    if total == 0:
        sentiment_score = 50.0
    else:
        # Neutral articles contribute a half-weight towards the midpoint
        sentiment_score = ((positive_count * 100) + (neutral_count * 50)) / total

    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "sentiment_score": round(sentiment_score, 2)
    }


def format_large_number(num):
    """Format large numbers (market cap) into readable strings like 2.5T, 800B, etc."""
    if num is None:
        return "N/A"
    try:
        num = float(num)
    except (ValueError, TypeError):
        return "N/A"

    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif abs_num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    else:
        return f"${num:,.2f}"


def format_percentage(value):
    """
    Format a fractional value (e.g. 0.015) as a percentage string.
    Some yfinance versions return dividend yield already as a percentage
    (e.g. 1.5 instead of 0.015), so values >= 1 are assumed to already be
    a percentage and are not multiplied by 100 again.
    """
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"

    if value >= 1:
        return f"{value:.2f}%"
    return f"{value * 100:.2f}%"
