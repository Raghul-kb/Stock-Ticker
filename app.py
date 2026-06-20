"""
app.py
Streamlit UI for the Stock Ticker Research Platform.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

from agents import ResearchAgent, RecommendationAgent
from pdf_generator import generate_pdf_report
from utils import DataFetchError, format_large_number, format_percentage

st.set_page_config(
    page_title="Stock Ticker Research Platform",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    <style>
    .rec-badge {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 22px;
        font-weight: 700;
        color: white;
        text-align: center;
    }
    .rec-buy { background-color: #1a8c1a; }
    .rec-sell { background-color: #c62828; }
    .rec-hold { background-color: #d4860b; }
    .news-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Stock Ticker Research Platform")

if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "recommendation_data" not in st.session_state:
    st.session_state.recommendation_data = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter Stock Ticker Symbol", placeholder="e.g. AAPL, MSFT, TSLA").strip().upper()
with col2:
    st.write("")
    st.write("")
    generate_clicked = st.button("🔍 Generate Report", use_container_width=True, type="primary")

if generate_clicked:
    if not ticker_input:
        st.error("Please enter a valid ticker symbol.")
    else:
        try:
            with st.spinner(f"Researching {ticker_input}... fetching data, indicators, and news"):
                research_agent = ResearchAgent(ticker_input)
                research_data = research_agent.run()
                st.session_state.research_data = research_data

            with st.spinner("Generating AI-powered recommendation..."):
                recommendation_agent = RecommendationAgent()
                recommendation_data = recommendation_agent.run(research_data)
                st.session_state.recommendation_data = recommendation_data

            st.session_state.pdf_path = None
            st.success(f"Report generated for {ticker_input}")

        except DataFetchError as e:
            st.error(f"⚠️ {str(e)}")
            st.session_state.research_data = None
            st.session_state.recommendation_data = None
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {str(e)}")
            st.session_state.research_data = None
            st.session_state.recommendation_data = None

research_data = st.session_state.research_data
recommendation_data = st.session_state.recommendation_data

if research_data and recommendation_data:
    stock_info = research_data["stock_info"]
    technicals = research_data["technicals"]
    sentiment = research_data["sentiment"]
    news = research_data["news"]

    st.divider()
    st.header(f"{stock_info.get('company_name', research_data['ticker'])} ({research_data['ticker']})")
    st.caption(f"Sector: {stock_info.get('sector', 'N/A')}  |  Industry: {stock_info.get('industry', 'N/A')}")

    st.subheader("📊 Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${stock_info.get('current_price', 'N/A')}")
    m2.metric("Market Cap", format_large_number(stock_info.get("market_cap")))
    m3.metric("PE Ratio", f"{stock_info.get('pe_ratio'):.2f}" if stock_info.get("pe_ratio") else "N/A")
    m4.metric("Dividend Yield", format_percentage(stock_info.get("dividend_yield")))

    m5, m6 = st.columns(2)
    m5.metric("52 Week High", f"${stock_info.get('fifty_two_week_high', 'N/A')}")
    m6.metric("52 Week Low", f"${stock_info.get('fifty_two_week_low', 'N/A')}")

    st.divider()

    st.subheader("📉 Technical Analysis")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("RSI (14)", technicals.get("rsi", "N/A"))
    t2.metric("MA 50", technicals.get("ma50", "N/A"))
    t3.metric("MA 200", technicals.get("ma200", "N/A"))
    t4.metric("Volatility (Annualized)", f"{technicals.get('volatility', 'N/A')}%")

    if technicals.get("ma50") and technicals.get("ma200"):
        trend = "🟢 Bullish (MA50 > MA200)" if technicals["ma50"] > technicals["ma200"] else "🔴 Bearish (MA50 < MA200)"
        st.info(f"**Trend Signal:** {trend}")

    st.divider()

    st.subheader("📰 News & Sentiment")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Positive", sentiment.get("positive_count", 0))
    s2.metric("Negative", sentiment.get("negative_count", 0))
    s3.metric("Neutral", sentiment.get("neutral_count", 0))
    s4.metric("Sentiment Score", f"{sentiment.get('sentiment_score', 0)}/100")

    st.progress(min(int(sentiment.get("sentiment_score", 0)), 100) / 100)

    if news:
        with st.expander(f"View {len(news)} Recent Headlines", expanded=False):
            for article in news[:10]:
                title = article.get("title", "")
                if not title:
                    continue
                source = article.get("source", "Unknown")
                url = article.get("url", "")
                published = article.get("published_at", "")[:10]
                st.markdown(
                    f"""<div class="news-card">
                    <b>{title}</b><br/>
                    <span style="color:gray;font-size:13px;">{source} &bull; {published}</span><br/>
                    <a href="{url}" target="_blank">Read more</a>
                    </div>""",
                    unsafe_allow_html=True
                )
    else:
        st.warning("No recent news articles found for this ticker.")

    st.divider()

    st.subheader("🤖 AI Recommendation")
    rec = recommendation_data.get("recommendation", "HOLD")
    badge_class = {"BUY": "rec-buy", "SELL": "rec-sell", "HOLD": "rec-hold"}.get(rec, "rec-hold")

    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        st.markdown(f'<div class="rec-badge {badge_class}">{rec}</div>', unsafe_allow_html=True)
        st.metric("Confidence Score", f"{recommendation_data.get('confidence_score', 'N/A')}/100")
        st.write(f"**Market Sentiment:** {recommendation_data.get('market_sentiment', 'N/A')}")
        st.caption(f"Rule-based baseline signal: {recommendation_data.get('rule_based_signal', 'N/A')}")
    with rcol2:
        st.write("**Rationale:**")
        st.write(recommendation_data.get("rationale", "No rationale available."))

    st.divider()

    st.subheader("📄 Export Report")
    if st.button("Generate PDF Report"):
        try:
            with st.spinner("Building PDF report..."):
                pdf_path = generate_pdf_report(research_data, recommendation_data)
                st.session_state.pdf_path = pdf_path
            st.success("PDF report generated successfully!")
        except Exception as e:
            st.error(f"⚠️ Failed to generate PDF: {str(e)}")

    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=f,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf"
            )

else:
    st.info("👆 Enter a ticker symbol and click **Generate Report** to begin your research.")