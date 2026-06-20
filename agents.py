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
    Collects raw stock data, computes technical indicators, fetches news,
    and calculates sentiment. Returns a structured dict for the next agent.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.strip().upper()

    def run(self) -> dict:
        if not self.ticker:
            raise DataFetchError("Ticker symbol cannot be empty.")

        stock_info = fetch_stock_info(self.ticker)
        history = stock_info.pop("history")

        technicals = calculate_technical_indicators(history)

        try:
            news_list = fetch_news(self.ticker, stock_info.get("company_name", ""))
        except DataFetchError:
            news_list = []

        sentiment = calculate_sentiment(news_list)

        structured_data = {
            "ticker": self.ticker,
            "stock_info": stock_info,
            "technicals": technicals,
            "news": news_list,
            "sentiment": sentiment
        }
        return structured_data


class RecommendationAgent:
    """
    Agent 2: Recommendation Agent
    Applies rule-based BUY/SELL/HOLD logic, then asks Groq's
    llama-3.3-70b-versatile for confidence score, sentiment label, rationale.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise DataFetchError("GROQ_API_KEY not found. Please set it in your .env file.")
        self.client = Groq(api_key=GROQ_API_KEY)

    @staticmethod
    def rule_based_signal(technicals: dict, sentiment: dict) -> str:
        """
        BUY:  RSI < 70  AND MA50 > MA200 AND Sentiment Score > 60
        SELL: RSI > 75  AND MA50 < MA200
        Otherwise: HOLD
        """
        rsi = technicals.get("rsi")
        ma50 = technicals.get("ma50")
        ma200 = technicals.get("ma200")
        sentiment_score = sentiment.get("sentiment_score")

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
        news = research_data["news"][:5]

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