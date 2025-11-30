import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis - Dual approach
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Stock Comparator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .strong-buy {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
    }
    .buy {
        background-color: #d1ecf1;
        border: 2px solid #bee5eb;
    }
    .hold {
        background-color: #fff3cd;
        border: 2px solid #ffeaa7;
    }
    .sell {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Default configuration
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEFAULT_WEIGHTS = {
    "annual_return": 0.18,
    "sharpe": 0.15,
    "volatility": 0.10,
    "pe": 0.07,
    "dividend_yield": 0.07,
    "momentum": 0.10,
    "sentiment": 0.18,
    "technical": 0.15
}

def fetch_data(ticker, period_years):
    end = datetime.today()
    start = end - timedelta(days=int(365.25 * period_years))
    tk = yf.Ticker(ticker)
    # history may throw or be empty; handle upstream
    hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    return hist, info, tk

def annualized_return_from_series(price_series):
    if price_series.empty:
        return np.nan
    n_days = (price_series.index[-1] - price_series.index[0]).days
    if n_days <= 0:
        return np.nan
    total_return = price_series.iloc[-1] / price_series.iloc[0] - 1
    years = n_days / 365.25
    try:
        return (1 + total_return) ** (1 / years) - 1
    except Exception:
        return np.nan

def annualized_volatility(daily_returns):
    return np.nan if daily_returns.dropna().empty else daily_returns.std() * np.sqrt(252)

def sharpe_ratio(annual_return, ann_vol, risk_free):
    if pd.isna(annual_return) or pd.isna(ann_vol) or ann_vol == 0:
        return np.nan
    return (annual_return - risk_free) / ann_vol

def momentum_12m(price_series):
    if price_series.empty:
        return np.nan
    end = price_series.index[-1]
    start_date = end - pd.DateOffset(months=12)
    try:
        start_price = price_series.loc[price_series.index >= start_date].iloc[0]
    except Exception:
        start_price = price_series.iloc[0]
    try:
        return price_series.iloc[-1] / start_price - 1
    except Exception:
        return np.nan

def safe_get(info, key, default=np.nan):
    try:
        val = info.get(key, default)
        if val is None:
            return default
        return val
    except Exception:
        return default

def get_comprehensive_sentiment(ticker_obj, ticker_symbol):
    sentiment_data = {
        'textblob_score': 0.0,
        'vader_score': 0.0,
        'combined_score': 0.0,
        'article_count': 0,
        'positive_count': 0,
        'negative_count': 0,
        'neutral_count': 0
    }

    if not TEXTBLOB_AVAILABLE and not VADER_AVAILABLE:
        return sentiment_data

    try:
        vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        # ticker_obj.news may raise or be attribute-less; guard it
        news = []
        try:
            news = getattr(ticker_obj, "news", []) or []
        except Exception:
            news = []
        if not news:
            return sentiment_data

        textblob_scores = []
        vader_scores = []

        for article in news[:15]:
            title = article.get('title', '') if isinstance(article, dict) else ''
            summary = article.get('summary', '') if isinstance(article, dict) else ''
            text = f"{title}. {summary}"

            if text.strip():
                if TEXTBLOB_AVAILABLE:
                    try:
                        blob = TextBlob(text)
                        tb_score = blob.sentiment.polarity
                        textblob_scores.append(tb_score)
                    except Exception:
                        pass

                if VADER_AVAILABLE and vader_analyzer:
                    try:
                        vader_result = vader_analyzer.polarity_scores(text)
                        vader_scores.append(vader_result['compound'])
                        if vader_result['compound'] >= 0.05:
                            sentiment_data['positive_count'] += 1
                        elif vader_result['compound'] <= -0.05:
                            sentiment_data['negative_count'] += 1
                        else:
                            sentiment_data['neutral_count'] += 1
                    except Exception:
                        pass

        if textblob_scores:
            sentiment_data['textblob_score'] = float(np.mean(textblob_scores))
        if vader_scores:
            sentiment_data['vader_score'] = float(np.mean(vader_scores))

        sentiment_data['article_count'] = len(textblob_scores) if textblob_scores else len(vader_scores)

        scores = []
        if TEXTBLOB_AVAILABLE and textblob_scores:
            scores.append(sentiment_data['textblob_score'])
        if VADER_AVAILABLE and vader_scores:
            scores.append(sentiment_data['vader_score'])

        if scores:
            sentiment_data['combined_score'] = float(np.mean(scores))

        return sentiment_data

    except Exception as e:
        # keep this non-fatal for the app
        st.warning(f"Sentiment analysis error for {ticker_symbol}: {e}")
        return sentiment_data

def compute_technical_indicators(hist_df):
    if not TALIB_AVAILABLE or hist_df.empty or len(hist_df) < 50:
        return {
            'rsi': np.nan,
            'macd_signal': np.nan,
            'bb_position': np.nan,
            'ma_trend': np.nan,
            'adx': np.nan,
            'stoch': np.nan,
            'technical_score': np.nan,
            'signal_strength': 'N/A'
        }

    try:
        close = hist_df['Close'].values
        high = hist_df['High'].values
        low = hist_df['Low'].values

        # RSI (14-day)
        rsi = talib.RSI(close, timeperiod=14)
        current_rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0

        # MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_signal = 1 if macd[-1] > signal[-1] else -1 if not np.isnan(macd[-1]) else 0

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        bb_position = float((close[-1] - lower[-1]) / (upper[-1] - lower[-1])) if not np.isnan(upper[-1]) and (upper[-1] - lower[-1]) != 0 else 0.5

        # Moving Averages
        ma50 = talib.SMA(close, timeperiod=50)
        ma200 = talib.SMA(close, timeperiod=200)
        ma_trend = 1 if ma50[-1] > ma200[-1] else -1 if not np.isnan(ma50[-1]) and not np.isnan(ma200[-1]) else 0

        # ADX
        adx = talib.ADX(high, low, close, timeperiod=14)
        current_adx = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0

        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close)
        current_stoch = float(slowk[-1]) if not np.isnan(slowk[-1]) else 50.0

        # Scoring logic
        rsi_score = 0.0
        if current_rsi < 30:
            rsi_score = 0.9
        elif current_rsi < 40:
            rsi_score = 0.7
        elif current_rsi < 60:
            rsi_score = 0.5
        elif current_rsi < 70:
            rsi_score = 0.3
        else:
            rsi_score = 0.1

        macd_score = 0.7 if macd_signal == 1 else 0.3
        bb_score = bb_position
        ma_score = 0.7 if ma_trend == 1 else 0.3
        adx_score = min(current_adx / 50, 1.0) if current_adx > 25 else 0.3

        stoch_score = 0.0
        if current_stoch < 20:
            stoch_score = 0.8
        elif current_stoch < 80:
            stoch_score = 0.5
        else:
            stoch_score = 0.2

        technical_score = (
            rsi_score * 0.25 +
            macd_score * 0.25 +
            bb_score * 0.15 +
            ma_score * 0.20 +
            adx_score * 0.10 +
            stoch_score * 0.05
        )

        if technical_score >= 0.7:
            signal_strength = "STRONG BUY"
        elif technical_score >= 0.6:
            signal_strength = "BUY"
        elif technical_score >= 0.4:
            signal_strength = "HOLD"
        elif technical_score >= 0.3:
            signal_strength = "WEAK SELL"
        else:
            signal_strength = "SELL"

        return {
            'rsi': current_rsi,
            'macd_signal': macd_signal,
            'bb_position': bb_position,
            'ma_trend': ma_trend,
            'adx': current_adx,
            'stoch': current_stoch,
            'technical_score': technical_score,
            'signal_strength': signal_strength
        }
    except Exception as e:
        st.warning(f"Technical analysis error: {e}")
        return {
            'rsi': np.nan,
            'macd_signal': np.nan,
            'bb_position': np.nan,
            'ma_trend': np.nan,
            'adx': np.nan,
            'stoch': np.nan,
            'technical_score': np.nan,
            'signal_strength': 'N/A'
        }

def build_metrics(tickers, history_years, risk_free):
    rows = []
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for i, t in enumerate(tickers):
        status_text.text(f"Processing {t}... ({i+1}/{len(tickers)})")
        progress_bar.progress((i + 1) / max(1, len(tickers)))

        try:
            hist, info, ticker_obj = fetch_data(t, history_years)
        except Exception as e:
            st.error(f"Failed to fetch {t}: {e}")
            hist, info, ticker_obj = pd.DataFrame(), {}, None

        if hist.empty:
            ann_return = np.nan
            ann_vol = np.nan
            sr = np.nan
            mom12 = np.nan
        else:
            close = hist["Close"]
            ann_return = annualized_return_from_series(close)
            daily_ret = close.pct_change().dropna()
            ann_vol = annualized_volatility(daily_ret)
            sr = sharpe_ratio(ann_return, ann_vol, risk_free)
            mom12 = momentum_12m(close)

        trailingPE = safe_get(info, "trailingPE", np.nan)
        div_yield = safe_get(info, "dividendYield", np.nan)
        shortName = safe_get(info, "shortName", t)
        sector = safe_get(info, "sector", "Unknown")
        marketCap = safe_get(info, "marketCap", np.nan)
        beta = safe_get(info, "beta", np.nan)

        sentiment_data = get_comprehensive_sentiment(ticker_obj, t) if ticker_obj else {
            'textblob_score': 0.0, 'vader_score': 0.0, 'combined_score': 0.0,
            'article_count': 0, 'positive_count': 0, 'negative_count': 0, 'neutral_count': 0
        }

        tech_indicators = compute_technical_indicators(hist)

        rows.append({
            "ticker": t,
            "name": shortName,
            "sector": sector,
            "marketCap": marketCap,
            "annual_return": ann_return,
            "volatility": ann_vol,
            "sharpe": sr,
            "momentum": mom12,
            "pe": trailingPE,
            "dividend_yield": div_yield,
            "beta": beta,
            "sentiment": sentiment_data['combined_score'],
            "textblob_sentiment": sentiment_data['textblob_score'],
            "vader_sentiment": sentiment_data['vader_score'],
            "news_articles": sentiment_data['article_count'],
            "positive_news": sentiment_data['positive_count'],
            "negative_news": sentiment_data['negative_count'],
            "neutral_news": sentiment_data['neutral_count'],
            "technical": tech_indicators['technical_score'],
            "rsi": tech_indicators['rsi'],
            "macd_signal": tech_indicators['macd_signal'],
            "bb_position": tech_indicators['bb_position'],
            "ma_trend": tech_indicators['ma_trend'],
            "adx": tech_indicators['adx'],
            "stoch": tech_indicators['stoch'],
            "tech_signal": tech_indicators['signal_strength']
        })

    progress_bar.empty()
    status_text.empty()

    df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
    return df

def normalize_series_for_score(s: pd.Series, higher_is_better=True):
    s_clean = s.copy().astype(float)
    mask = ~s_clean.isna()
    if mask.sum() == 0:
        return pd.Series(0.0, index=s.index)
    vals = s_clean[mask]
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        norm = pd.Series(0.5, index=s.index)
    else:
        norm = (s_clean - lo) / (hi - lo)
    norm = norm.fillna(0.0)
    if not higher_is_better:
        norm = 1.0 - norm
    return norm

def score_universe(df, weights):
    if df.empty:
        return df
    nr = pd.DataFrame(index=df.index)
    nr["annual_return"] = normalize_series_for_score(df.get("annual_return", pd.Series(dtype=float)), higher_is_better=True)
    nr["sharpe"] = normalize_series_for_score(df.get("sharpe", pd.Series(dtype=float)), higher_is_better=True)
    nr["volatility"] = normalize_series_for_score(df.get("volatility", pd.Series(dtype=float)), higher_is_better=False)
    nr["pe"] = normalize_series_for_score(df.get("pe", pd.Series(dtype=float)), higher_is_better=False)
    nr["dividend_yield"] = normalize_series_for_score(df.get("dividend_yield", pd.Series(dtype=float)), higher_is_better=True)
    nr["momentum"] = normalize_series_for_score(df.get("momentum", pd.Series(dtype=float)), higher_is_better=True)
    nr["sentiment"] = normalize_series_for_score(df.get("sentiment", pd.Series(dtype=float)), higher_is_better=True)
    nr["technical"] = normalize_series_for_score(df.get("technical", pd.Series(dtype=float)), higher_is_better=True)

    scores = pd.Series(0.0, index=df.index)
    for metric, w in weights.items():
        if metric not in nr.columns:
            continue
        scores += nr[metric] * w

    result = df.copy()
    for col in nr.columns:
        result[f"norm_{col}"] = nr[col]
    result["composite_score"] = scores
    result = result.sort_values("composite_score", ascending=False)
    return result

def display_final_recommendation(result_df):
    if result_df.empty:
        st.error("No results to recommend.")
        return

    top = result_df.iloc[0]
    ticker = top.name if hasattr(top, "name") else result_df.index[0]

    # Calculate confidence level
    confidence_factors = []

    if len(result_df) > 1:
        second_score = result_df.iloc[1]['composite_score']
        score_gap = top['composite_score'] - second_score
        if score_gap > 0.15:
            confidence_factors.append(("Clear leader", 0.3))
        elif score_gap > 0.08:
            confidence_factors.append(("Moderate lead", 0.2))
        else:
            confidence_factors.append(("Tight race", 0.1))

    sentiment = top.get('sentiment', 0)
    if sentiment > 0.15:
        confidence_factors.append(("Strong positive sentiment", 0.25))
    elif sentiment > 0:
        confidence_factors.append(("Mild positive sentiment", 0.15))
    elif sentiment > -0.15:
        confidence_factors.append(("Neutral sentiment", 0.05))
    else:
        confidence_factors.append(("Negative sentiment", -0.1))

    tech_score = top.get('technical', np.nan)
    if not pd.isna(tech_score):
        if tech_score > 0.65:
            confidence_factors.append(("Strong technical buy signals", 0.25))
        elif tech_score > 0.5:
            confidence_factors.append(("Positive technical signals", 0.15))
        else:
            confidence_factors.append(("Weak technical signals", 0.05))

    if not pd.isna(top.get('sharpe', np.nan)) and top.get('sharpe', 0) > 1.0:
        confidence_factors.append(("Strong risk-adjusted returns", 0.2))
    elif not pd.isna(top.get('sharpe', np.nan)) and top.get('sharpe', 0) > 0.5:
        confidence_factors.append(("Good risk-adjusted returns", 0.1))

    base_confidence = 0.5
    confidence_score = base_confidence + sum(factor[1] for factor in confidence_factors)
    confidence_score = min(max(confidence_score, 0), 1.0)

    if confidence_score >= 0.8:
        recommendation = "STRONG BUY"
        css_class = "strong-buy"
        emoji = "🟢🟢🟢"
    elif confidence_score >= 0.65:
        recommendation = "BUY"
        css_class = "buy"
        emoji = "🟢🟢"
    elif confidence_score >= 0.5:
        recommendation = "MODERATE BUY"
        css_class = "buy"
        emoji = "🟢"
    elif confidence_score >= 0.35:
        recommendation = "HOLD"
        css_class = "hold"
        emoji = "🟡"
    else:
        recommendation = "CONSIDER ALTERNATIVES"
        css_class = "sell"
        emoji = "🔴"

    # Display recommendation
    st.markdown(f"""
    <div class="recommendation-box {css_class}">
        <h2>{emoji} {recommendation} - {ticker}</h2>
        <h3>{top.get('name', '')}</h3>
        <p><strong>Composite Score:</strong> {top['composite_score']:.3f}/1.000</p>
        <p><strong>Confidence Level:</strong> {confidence_score:.1%}</p>
        <p><strong>Sector:</strong> {top.get('sector', 'Unknown')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 Performance")
        if not pd.isna(top.get('annual_return', np.nan)):
            st.metric("Annual Return", f"{top['annual_return']:.2%}")
        if not pd.isna(top.get('sharpe', np.nan)):
            st.metric("Sharpe Ratio", f"{top['sharpe']:.2f}")
        if not pd.isna(top.get('volatility', np.nan)):
            st.metric("Volatility", f"{top['volatility']:.2%}")
        if not pd.isna(top.get('momentum', np.nan)):
            st.metric("12-Month Momentum", f"{top['momentum']:.2%}")

    with col2:
        st.subheader("💰 Valuation & Sentiment")
        if not pd.isna(top.get('pe', np.nan)):
            st.metric("P/E Ratio", f"{top['pe']:.2f}")
        if not pd.isna(top.get('dividend_yield', np.nan)) and top.get('dividend_yield', 0) > 0:
            st.metric("Dividend Yield", f"{top['dividend_yield']:.2%}")
        st.metric("Sentiment Score", f"{top.get('sentiment', 0):.3f}")
        if top.get('news_articles', 0) > 0:
            st.metric("News Articles", int(top.get('news_articles', 0)))

    with col3:
        st.subheader("📊 Technical Analysis")
        if not pd.isna(top.get('rsi', np.nan)):
            rsi_status = "Oversold 🟢" if top['rsi'] < 30 else "Overbought 🔴" if top['rsi'] > 70 else "Neutral 🟡"
            st.metric("RSI (14)", f"{top['rsi']:.1f}", rsi_status)
        if not pd.isna(top.get('technical', np.nan)):
            st.metric("Technical Score", f"{top['technical']:.3f}")
        st.metric("Technical Signal", top.get('tech_signal', 'N/A'))
        if not pd.isna(top.get('adx', np.nan)):
            st.metric("Trend Strength (ADX)", f"{top['adx']:.1f}")

def main():
    st.markdown('<h1 class="main-header">📈 Stock Comparator Pro</h1>', unsafe_allow_html=True)
    st.markdown("Comprehensive stock analysis with fundamentals, sentiment analysis, and technical indicators")

    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")

        # Ticker input
        ticker_input = st.text_area(
            "Stock Tickers (one per line)",
            value="\n".join(DEFAULT_TICKERS),
            height=100
        )
        tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]

        # Analysis parameters
        history_years = st.slider("Analysis Period (Years)", 1, 10, 3)
        risk_free = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0) / 100

        st.subheader("Scoring Weights")
        weights = {}
        for metric, default_weight in DEFAULT_WEIGHTS.items():
            weights[metric] = st.slider(
                f"{metric.replace('_', ' ').title()}",
                0.0, 0.3, default_weight,
                0.01
            )

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for metric in weights:
                weights[metric] = weights[metric] / total_weight

        st.info(f"Total weight: {sum(weights.values()):.0%}")

        # Analysis capabilities
        st.subheader("Analysis Capabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"TextBlob: {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
            st.write(f"VADER: {'✅' if VADER_AVAILABLE else '❌'}")
        with col2:
            st.write(f"TA-Lib: {'✅' if TALIB_AVAILABLE else '❌'}")

        analyze_clicked = st.button("🚀 Analyze Stocks", type="primary")

    # If button not pressed, stop the app (sidebar still visible)
    if not analyze_clicked:
        st.stop()

    # Analysis
    with st.spinner("Fetching stock data and performing analysis..."):
        df = build_metrics(tickers, history_years, risk_free)

    if df.empty:
        st.error("No data could be fetched for the provided tickers. Please check the ticker symbols and try again.")
        return

    scored_df = score_universe(df, weights)

    # Display results
    st.header("📊 Ranked Stock Comparison")

    # Main results table
    display_columns = [
        "name", "sector", "annual_return", "sharpe", "volatility",
        "sentiment", "technical", "composite_score"
    ]
    # safely pick available columns
    display_df = scored_df[[c for c in display_columns if c in scored_df.columns]].copy()

    # Format percentages
    percent_columns = ["annual_return", "volatility", "momentum"]
    for col in percent_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")

    # Format other numeric columns
    numeric_columns = ["sharpe", "sentiment", "technical", "composite_score"]
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")

    st.dataframe(display_df, use_container_width=True)

    # Final recommendation
    st.header("🎯 Investment Recommendation")
    display_final_recommendation(scored_df)

    # Detailed analysis expanders
    st.header("🔍 Detailed Analysis")

    for ticker in scored_df.index[:3]:  # Show details for top 3
        with st.expander(f"Detailed Analysis: {ticker}"):
            stock_data = scored_df.loc[ticker]
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Fundamentals")
                if not pd.isna(stock_data.get('marketCap', np.nan)):
                    try:
                        st.write(f"Market Cap: ${int(stock_data['marketCap']):,}")
                    except Exception:
                        st.write(f"Market Cap: {stock_data.get('marketCap')}")
                if not pd.isna(stock_data.get('beta', np.nan)):
                    st.write(f"Beta: {stock_data['beta']:.2f}")
                if not pd.isna(stock_data.get('pe', np.nan)):
                    st.write(f"P/E Ratio: {stock_data['pe']:.2f}")
                if not pd.isna(stock_data.get('dividend_yield', np.nan)) and stock_data.get('dividend_yield', 0) > 0:
                    st.write(f"Dividend Yield: {stock_data['dividend_yield']:.2%}")

            with col2:
                st.subheader("Sentiment Analysis")
                if stock_data.get('news_articles', 0) > 0:
                    st.write(f"Articles Analyzed: {int(stock_data['news_articles'])}")
                    st.write(f"Positive: {int(stock_data.get('positive_news',0))} | "
                           f"Neutral: {int(stock_data.get('neutral_news',0))} | "
                           f"Negative: {int(stock_data.get('negative_news',0))}")
                    if TEXTBLOB_AVAILABLE:
                        st.write(f"TextBlob Score: {stock_data.get('textblob_sentiment', 0):.3f}")
                    if VADER_AVAILABLE:
                        st.write(f"VADER Score: {stock_data.get('vader_sentiment', 0):.3f}")
                    st.write(f"Combined Sentiment: {stock_data.get('sentiment', 0):.3f}")
                else:
                    st.write("No sentiment data available")

    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Disclaimer:** This analysis is for informational purposes only and should not be considered 
    as financial advice. Always conduct your own research and consult with a qualified financial 
    advisor before making investment decisions. Past performance does not guarantee future results.
    """)

if __name__ == "__main__":
    main()
