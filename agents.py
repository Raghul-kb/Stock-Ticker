"""
agents.py
Defines two agents:
  1. ResearchAgent  - collects stock data, technical indicators, news, sentiment.
  2. RecommendationAgent - uses rule-based logic + Groq LLM to generate a
     BUY / SELL / HOLD recommendation with confidence score and rationale.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from utils import (
    fetch_stock_info,
    calculate_technical_indicators,
    fetch_news,
    calculate_sentiment,
    DataFetchError
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


class ResearchAgent:
    """
    Agent 1: Research Agent
    Responsible for collecting raw stock data, computing technical
    indicators, fetching relevant news, and calculating sentiment.
    Returns a single structured dictionary consumed by the
    RecommendationAgent.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.strip().upper()

    def run(self) -> dict:
        """Execute the full research pipeline and return structured data."""
        if not self.ticker:
            raise DataFetchError("Ticker symbol cannot be empty.")

        # Step 1: Stock info
        stock_info = fetch_stock_info(self.ticker)
        history = stock_info.pop("history")

        # Step 2: Technical indicators
        technicals = calculate_technical_indicators(history)

        # Step 3: News
        try:
            news_list = fetch_news(self.ticker, stock_info.get("company_name", ""))
        except DataFetchError:
            # News failures shouldn't crash the whole pipeline; fall back to empty list
            news_list = []

        # Step 4: Sentiment
        sentiment = calculate_sentiment(news_list)

        structured_data = {
            "ticker": self.ticker,
            "stock_info": stock_info,
            "technicals": technicals,
            "news": news_list,
            "sentiment": sentiment,
            "history": history
        }
        return structured_data


class RecommendationAgent:
    """
    Agent 2: Recommendation Agent
    Consumes the structured output of ResearchAgent, applies a rule-based
    BUY/SELL/HOLD check, and asks the Groq LLM (llama-3.3-70b-versatile)
    to produce a confidence score, market sentiment label, and a short
    rationale grounded in the provided data.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise DataFetchError("GROQ_API_KEY not found. Please set it in your .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)

    @staticmethod
    def rule_based_signal(technicals: dict, sentiment: dict) -> str:
        """
        Apply the simple deterministic recommendation logic:

        BUY:  RSI < 70  AND MA50 > MA200 AND Sentiment Score > 60
        SELL: RSI > 75  AND MA50 < MA200
        Otherwise: HOLD
        """
        rsi = technicals.get("rsi")
        ma50 = technicals.get("ma50")
        ma200 = technicals.get("ma200")
        sentiment_score = sentiment.get("sentiment_score")

        # If we don't have enough data to evaluate the rules, default to HOLD
        if rsi is None or ma50 is None or ma200 is None or sentiment_score is None:
            return "HOLD"

        if rsi < 70 and ma50 > ma200 and sentiment_score > 60:
            return "BUY"
        elif rsi > 75 and ma50 < ma200:
            return "SELL"
        else:
            return "HOLD"

    def _build_prompt(self, research_data: dict, rule_signal: str) -> str:
        stock_info = research_data["stock_info"]
        technicals = research_data["technicals"]
        sentiment = research_data["sentiment"]
        news = research_data["news"][:5]  # limit to top 5 headlines for prompt size

        news_summary = "\n".join(
            [f"- {n['title']} ({n['source']})" for n in news if n.get("title")]
        ) or "No recent news available."

        prompt = f"""You are a financial analyst assistant. Analyze the following stock data
and produce a recommendation. A rule-based system has already computed a baseline
signal of "{rule_signal}" using this logic:
- BUY: RSI < 70 AND MA50 > MA200 AND Sentiment Score > 60
- SELL: RSI > 75 AND MA50 < MA200
- Otherwise: HOLD

You may agree with this baseline signal or adjust it slightly if the data strongly
suggests otherwise, but stay close to the rule-based logic unless there is a clear reason.

STOCK DATA:
Ticker: {research_data['ticker']}
Company: {stock_info.get('company_name')}
Current Price: {stock_info.get('current_price')}
Market Cap: {stock_info.get('market_cap')}
PE Ratio: {stock_info.get('pe_ratio')}
Dividend Yield: {stock_info.get('dividend_yield')}
52 Week High: {stock_info.get('fifty_two_week_high')}
52 Week Low: {stock_info.get('fifty_two_week_low')}

TECHNICAL INDICATORS:
RSI (14): {technicals.get('rsi')}
50 Day MA: {technicals.get('ma50')}
200 Day MA: {technicals.get('ma200')}
Annualized Volatility: {technicals.get('volatility')}%

NEWS SENTIMENT:
Positive Articles: {sentiment.get('positive_count')}
Negative Articles: {sentiment.get('negative_count')}
Neutral Articles: {sentiment.get('neutral_count')}
Sentiment Score (0-100): {sentiment.get('sentiment_score')}

RECENT HEADLINES:
{news_summary}

Respond ONLY with a valid JSON object (no markdown, no commentary) in this exact format:
{{
  "recommendation": "BUY/SELL/HOLD",
  "confidence_score": <integer 0-100>,
  "market_sentiment": "Bullish/Bearish/Neutral",
  "rationale": "<2-3 sentence rationale grounded in the data above>"
}}
"""
        return prompt

    def run(self, research_data: dict) -> dict:
        """
        Generate the final recommendation by combining the rule-based
        signal with an LLM-generated confidence score, sentiment label,
        and rationale.
        """
        technicals = research_data["technicals"]
        sentiment = research_data["sentiment"]

        rule_signal = self.rule_based_signal(technicals, sentiment)
        prompt = self._build_prompt(research_data, rule_signal)

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise financial analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            raw_text = response.choices[0].message.content.strip()

            # Clean up potential markdown code fences
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            llm_result = json.loads(cleaned)

            recommendation = llm_result.get("recommendation", rule_signal).upper()
            if recommendation not in ("BUY", "SELL", "HOLD"):
                recommendation = rule_signal

            result = {
                "recommendation": recommendation,
                "confidence_score": llm_result.get("confidence_score", 50),
                "market_sentiment": llm_result.get("market_sentiment", "Neutral"),
                "rationale": llm_result.get("rationale", "No rationale provided."),
                "rule_based_signal": rule_signal
            }
            return result

        except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
            # Fallback: if the LLM response can't be parsed, use rule-based signal
            return {
                "recommendation": rule_signal,
                "confidence_score": 50,
                "market_sentiment": "Neutral",
                "rationale": f"LLM response could not be parsed; falling back to rule-based signal. ({str(e)})",
                "rule_based_signal": rule_signal
            }
        except Exception as e:
            return {
                "recommendation": rule_signal,
                "confidence_score": 50,
                "market_sentiment": "Neutral",
                "rationale": f"Groq API call failed; falling back to rule-based signal. Error: {str(e)}",
                "rule_based_signal": rule_signal
            }
