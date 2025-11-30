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
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Indian Stock Comparator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Indian Stock Sectors
INDIAN_STOCK_SECTORS = {
    "Information Technology (IT) & Services": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"
    ],
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "BAJFINANCE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS"
    ],
    "Conglomerates & Industrial": [
        "RELIANCE.NS", "LT.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "M&M.NS"
    ],
    "Consumer Goods & Telecom": [
        "ITC.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "BHARTIARTL.NS", "MARUTI.NS"
    ],
    "Energy & Commodities": [
        "ONGC.NS", "NTPC.NS", "COALINDIA.NS", "HINDALCO.NS", "JSWSTEEL.NS"
    ]
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sector-header {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
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
    .stock-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Default configuration
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
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
        info = tk.info
        return hist, info, tk
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), {}, None

def annualized_return_from_series(price_series):
    if price_series.empty:
        return np.nan
    n_days = (price_series.index[-1] - price_series.index[0]).days
    if n_days <= 0:
        return np.nan
    total_return = price_series.iloc[-1] / price_series.iloc[0] - 1
    years = n_days / 365.25
    return (1 + total_return) ** (1 / years) - 1

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
    return price_series.iloc[-1] / start_price - 1

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
        news = ticker_obj.news
        if not news or len(news) == 0:
            return sentiment_data
        
        textblob_scores = []
        vader_scores = []
        
        for article in news[:15]:
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title}. {summary}"
            
            if text.strip():
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    tb_score = blob.sentiment.polarity
                    textblob_scores.append(tb_score)
                    
                if VADER_AVAILABLE and vader_analyzer:
                    vader_result = vader_analyzer.polarity_scores(text)
                    vader_scores.append(vader_result['compound'])
                    
                    if vader_result['compound'] >= 0.05:
                        sentiment_data['positive_count'] += 1
                    elif vader_result['compound'] <= -0.05:
                        sentiment_data['negative_count'] += 1
                    else:
                        sentiment_data['neutral_count'] += 1
        
        if textblob_scores:
            sentiment_data['textblob_score'] = np.mean(textblob_scores)
        if vader_scores:
            sentiment_data['vader_score'] = np.mean(vader_scores)
        
        sentiment_data['article_count'] = len(textblob_scores) if textblob_scores else len(vader_scores)
        
        scores = []
        if TEXTBLOB_AVAILABLE and textblob_scores:
            scores.append(sentiment_data['textblob_score'])
        if VADER_AVAILABLE and vader_scores:
            scores.append(sentiment_data['vader_score'])
        
        if scores:
            sentiment_data['combined_score'] = np.mean(scores)
        
        return sentiment_data
        
    except Exception as e:
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
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
        
        # MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_signal = 1 if macd[-1] > signal[-1] else -1 if not np.isnan(macd[-1]) else 0
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        bb_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1]) if not np.isnan(upper[-1]) else 0.5
        
        # Moving Averages
        ma50 = talib.SMA(close, timeperiod=50)
        ma200 = talib.SMA(close, timeperiod=200)
        ma_trend = 1 if ma50[-1] > ma200[-1] else -1 if not np.isnan(ma50[-1]) and not np.isnan(ma200[-1]) else 0
        
        # ADX
        adx = talib.ADX(high, low, close, timeperiod=14)
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close)
        current_stoch = slowk[-1] if not np.isnan(slowk[-1]) else 50
        
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
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(tickers):
        status_text.text(f"Processing {t}... ({i+1}/{len(tickers)})")
        progress_bar.progress((i + 1) / len(tickers))
        
        hist, info, ticker_obj = fetch_data(t, history_years)
        
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
    
    df = pd.DataFrame(rows).set_index("ticker")
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
    nr = pd.DataFrame(index=df.index)
    nr["annual_return"] = normalize_series_for_score(df["annual_return"], higher_is_better=True)
    nr["sharpe"] = normalize_series_for_score(df["sharpe"], higher_is_better=True)
    nr["volatility"] = normalize_series_for_score(df["volatility"], higher_is_better=False)
    nr["pe"] = normalize_series_for_score(df["pe"], higher_is_better=False)
    nr["dividend_yield"] = normalize_series_for_score(df["dividend_yield"], higher_is_better=True)
    nr["momentum"] = normalize_series_for_score(df["momentum"], higher_is_better=True)
    nr["sentiment"] = normalize_series_for_score(df["sentiment"], higher_is_better=True)
    nr["technical"] = normalize_series_for_score(df["technical"], higher_is_better=True)
    
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

def display_sector_analysis(scored_df):
    """Display sector-wise analysis"""
    st.header("🏢 Sector-wise Analysis")
    
    # Group by sector and calculate average scores
    sector_data = []
    for sector_name, tickers in INDIAN_STOCK_SECTORS.items():
        sector_stocks = scored_df[scored_df.index.isin(tickers)]
        if not sector_stocks.empty:
            sector_data.append({
                'Sector': sector_name,
                'Stocks Analyzed': len(sector_stocks),
                'Avg Composite Score': sector_stocks['composite_score'].mean(),
                'Top Stock': sector_stocks.iloc[0].name,
                'Top Stock Score': sector_stocks.iloc[0]['composite_score']
            })
    
    sector_df = pd.DataFrame(sector_data).sort_values('Avg Composite Score', ascending=False)
    
    # Display sector performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sector Rankings")
        for i, row in sector_df.iterrows():
            st.markdown(f"""
            <div class="stock-card">
                <h4>{row['Sector']}</h4>
                <p><strong>Avg Score:</strong> {row['Avg Composite Score']:.3f}</p>
                <p><strong>Top Stock:</strong> {row['Top Stock']} ({row['Top Stock Score']:.3f})</p>
                <p><strong>Stocks Analyzed:</strong> {row['Stocks Analyzed']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📈 Sector Performance Heatmap")
        # Create a simple heatmap visualization
        heatmap_data = []
        for sector_name, tickers in INDIAN_STOCK_SECTORS.items():
            sector_stocks = scored_df[scored_df.index.isin(tickers)]
            if not sector_stocks.empty:
                for ticker in tickers:
                    if ticker in scored_df.index:
                        heatmap_data.append({
                            'Sector': sector_name,
                            'Stock': ticker,
                            'Score': scored_df.loc[ticker, 'composite_score']
                        })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            pivot_df = heatmap_df.pivot(index='Sector', columns='Stock', values='Score')
            st.dataframe(pivot_df.style.background_gradient(cmap='RdYlGn', axis=None), 
                        use_container_width=True)

def display_final_recommendation(result_df):
    if result_df.empty:
        st.error("No results to recommend.")
        return
    
    top = result_df.iloc[0]
    ticker = top.name
    
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
    
    sentiment = top['sentiment']
    if sentiment > 0.15:
        confidence_factors.append(("Strong positive sentiment", 0.25))
    elif sentiment > 0:
        confidence_factors.append(("Mild positive sentiment", 0.15))
    elif sentiment > -0.15:
        confidence_factors.append(("Neutral sentiment", 0.05))
    else:
        confidence_factors.append(("Negative sentiment", -0.1))
    
    tech_score = top['technical']
    if not pd.isna(tech_score):
        if tech_score > 0.65:
            confidence_factors.append(("Strong technical buy signals", 0.25))
        elif tech_score > 0.5:
            confidence_factors.append(("Positive technical signals", 0.15))
        else:
            confidence_factors.append(("Weak technical signals", 0.05))
    
    if not pd.isna(top['sharpe']) and top['sharpe'] > 1.0:
        confidence_factors.append(("Strong risk-adjusted returns", 0.2))
    elif not pd.isna(top['sharpe']) and top['sharpe'] > 0.5:
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
        if not pd.isna(top['annual_return']):
            st.metric("Annual Return", f"{top['annual_return']:.2%}")
        if not pd.isna(top['sharpe']):
            st.metric("Sharpe Ratio", f"{top['sharpe']:.2f}")
        if not pd.isna(top['volatility']):
            st.metric("Volatility", f"{top['volatility']:.2%}")
        if not pd.isna(top['momentum']):
            st.metric("12-Month Momentum", f"{top['momentum']:.2%}")
    
    with col2:
        st.subheader("💰 Valuation & Sentiment")
        if not pd.isna(top['pe']):
            st.metric("P/E Ratio", f"{top['pe']:.2f}")
        if not pd.isna(top['dividend_yield']) and top['dividend_yield'] > 0:
            st.metric("Dividend Yield", f"{top['dividend_yield']:.2%}")
        st.metric("Sentiment Score", f"{top['sentiment']:.3f}")
        if top['news_articles'] > 0:
            st.metric("News Articles", int(top['news_articles']))
    
    with col3:
        st.subheader("📊 Technical Analysis")
        if not pd.isna(top['rsi']):
            rsi_status = "Oversold 🟢" if top['rsi'] < 30 else "Overbought 🔴" if top['rsi'] > 70 else "Neutral 🟡"
            st.metric("RSI (14)", f"{top['rsi']:.1f}", rsi_status)
        if not pd.isna(top['technical']):
            st.metric("Technical Score", f"{top['technical']:.3f}")
        st.metric("Technical Signal", top.get('tech_signal', 'N/A'))
        if not pd.isna(top['adx']):
            st.metric("Trend Strength (ADX)", f"{top['adx']:.1f}")

def main():
    st.markdown('<h1 class="main-header">🇮🇳 Indian Stock Comparator Pro</h1>', unsafe_allow_html=True)
    st.markdown("Comprehensive analysis of Indian stocks with sector-wise insights")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Stock selection method
        selection_method = st.radio(
            "Select Stocks By:",
            ["Sector-wise", "Custom Selection"]
        )
        
        tickers = []
        
        if selection_method == "Sector-wise":
            st.subheader("🏢 Select Sectors")
            selected_sectors = []
            for sector_name, sector_tickers in INDIAN_STOCK_SECTORS.items():
                if st.checkbox(sector_name, value=True):
                    selected_sectors.extend(sector_tickers)
                    st.info(f"📊 {len(sector_tickers)} stocks in {sector_name}")
            
            tickers = selected_sectors
            
            # Also show individual stock selection within sectors
            st.subheader("🎯 Fine-tune Selection")
            for sector_name, sector_tickers in INDIAN_STOCK_SECTORS.items():
                with st.expander(f"{sector_name} Stocks"):
                    for ticker in sector_tickers:
                        if st.checkbox(ticker, value=True, key=ticker):
                            if ticker not in tickers:
                                tickers.append(ticker)
                        else:
                            if ticker in tickers:
                                tickers.remove(ticker)
        
        else:  # Custom Selection
            st.subheader("📝 Custom Stock Selection")
            ticker_input = st.text_area(
                "Enter stock tickers (one per line, use .NS for NSE)",
                value="RELIANCE.NS\nTCS.NS\nHDFCBANK.NS\nINFY.NS\nITC.NS",
                height=150
            )
            tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
        
        # Remove duplicates and ensure .NS suffix
        tickers = list(set(tickers))
        tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in tickers]
        
        st.success(f"🎯 Selected {len(tickers)} stocks for analysis")
        
        # Analysis parameters
        st.subheader("📅 Analysis Parameters")
        history_years = st.slider("Analysis Period (Years)", 1, 10, 3)
        risk_free = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
        
        st.subheader("⚖️ Scoring Weights")
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
        
        st.info(f"📊 Total weight: {sum(weights.values()):.0%}")
        
        # Analysis capabilities
        st.subheader("🔬 Analysis Capabilities")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"TextBlob: {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
            st.write(f"VADER: {'✅' if VADER_AVAILABLE else '❌'}")
        with col2:
            st.write(f"TA-Lib: {'✅' if TALIB_AVAILABLE else '❌'}")
        
        if st.button("🚀 Analyze Stocks", type="primary", use_container_width=True):
            return tickers, history_years, risk_free, weights
        else:
            st.stop()
    
    # Get parameters from sidebar
    tickers, history_years, risk_free, weights = main()
    
    if not tickers:
        st.error("❌ Please select at least one stock to analyze.")
        return
    
    # Analysis
    with st.spinner("🔄 Fetching stock data and performing analysis..."):
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
    display_df = scored_df[display_columns].copy()
    
    # Format percentages
    percent_columns = ["annual_return", "volatility"]
    for col in percent_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
    
    # Format other numeric columns
    numeric_columns = ["sharpe", "sentiment", "technical", "composite_score"]
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Sector Analysis
    display_sector_analysis(scored_df)
    
    # Final recommendation
    st.header("🎯 Top Investment Recommendation")
    display_final_recommendation(scored_df)
    
    # Detailed analysis expanders for top 5 stocks
    st.header("🔍 Detailed Stock Analysis")
    
    top_5_stocks = scored_df.head(5)
    for ticker in top_5_stocks.index:
        with st.expander(f"📋 Detailed Analysis: {ticker} - {scored_df.loc[ticker, 'name']}"):
            stock_data = scored_df.loc[ticker]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📈 Fundamentals")
                metrics = [
                    ("Annual Return", f"{stock_data['annual_return']:.2%}"),
                    ("Sharpe Ratio", f"{stock_data['sharpe']:.2f}"),
                    ("Volatility", f"{stock_data['volatility']:.2%}"),
                    ("Momentum (12M)", f"{stock_data['momentum']:.2%}")
                ]
                for label, value in metrics:
                    st.metric(label, value)
                
                if not pd.isna(stock_data['marketCap']):
                    st.metric("Market Cap", f"₹{stock_data['marketCap']:,.0f}")
                if not pd.isna(stock_data['beta']):
                    st.metric("Beta", f"{stock_data['beta']:.2f}")
            
            with col2:
                st.subheader("💰 Valuation")
                if not pd.isna(stock_data['pe']):
                    st.metric("P/E Ratio", f"{stock_data['pe']:.2f}")
                if not pd.isna(stock_data['dividend_yield']) and stock_data['dividend_yield'] > 0:
                    st.metric("Dividend Yield", f"{stock_data['dividend_yield']:.2%}")
                
                st.subheader("📰 Sentiment")
                if stock_data['news_articles'] > 0:
                    st.metric("Articles Analyzed", int(stock_data['news_articles']))
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        st.metric("😊 Positive", int(stock_data['positive_news']))
                    with col2b:
                        st.metric("😐 Neutral", int(stock_data['neutral_news']))
                    with col2c:
                        st.metric("😞 Negative", int(stock_data['negative_news']))
                    if TEXTBLOB_AVAILABLE:
                        st.metric("TextBlob Score", f"{stock_data['textblob_sentiment']:.3f}")
                    if VADER_AVAILABLE:
                        st.metric("VADER Score", f"{stock_data['vader_sentiment']:.3f}")
                else:
                    st.write("No sentiment data available")
            
            with col3:
                st.subheader("📊 Technical Analysis")
                tech_metrics = [
                    ("RSI (14)", f"{stock_data['rsi']:.1f}"),
                    ("ADX", f"{stock_data['adx']:.1f}"),
                    ("Stochastic", f"{stock_data['stoch']:.1f}")
                ]
                for label, value in metrics:
                    st.metric(label, value)
                
                st.metric("Technical Score", f"{stock_data['technical']:.3f}")
                st.metric("Signal", stock_data.get('tech_signal', 'N/A'))
                
                if not pd.isna(stock_data['macd_signal']):
                    macd_status = "Bullish 🟢" if stock_data['macd_signal'] > 0 else "Bearish 🔴"
                    st.metric("MACD", macd_status)
                if not pd.isna(stock_data['ma_trend']):
                    ma_status = "Bullish 🟢" if stock_data['ma_trend'] > 0 else "Bearish 🔴"
                    st.metric("MA Trend", ma_status)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **Disclaimer:** This analysis is for informational purposes only and should not be considered 
    as financial advice. Always conduct your own research and consult with a qualified financial 
    advisor before making investment decisions. Past performance does not guarantee future results.
    
    **Note:** Indian stock data is sourced from Yahoo Finance via yfinance library. 
    Ensure you use the correct ticker symbols with '.NS' suffix for NSE stocks.
    """)

if __name__ == "__main__":
    main()
