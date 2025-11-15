# main.py — Streamlit Stock Price Prediction App (LSTM + Indicators + Advanced Scoring)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import ta

# ================================
# INVESTMENT SCORING SYSTEM
# ================================
def calculate_investment_score(data):
    """
    Calculates a full weighted investment score (0–100) from 8 categories:
    Financial, Valuation, Growth, Risk, Momentum, Liquidity, Management, Industry.
    """

    # ------------------------------------------------------------
    # 1) NORMALIZATION FUNCTIONS (Convert raw values to 0–100 scale)
    # ------------------------------------------------------------

    def normalize_positive(val, max_val, min_val=0):
        """Higher value = better."""
        try:
            return max(0, min(100, (val - min_val) / (max_val - min_val) * 100))
        except:
            return 50  # neutral fallback

    def normalize_negative(val, max_val, min_val=0):
        """Lower value = better."""
        try:
            return max(0, min(100, (max_val - val) / (max_val - min_val) * 100))
        except:
            return 50

    def clamp(value):
        """Clamp value between 0 and 100."""
        return max(0, min(100, value))

    # ------------------------------------------------------------
    # 2) CATEGORY SCORES
    # ------------------------------------------------------------

    # ------------- FINANCIAL PERFORMANCE (20%) -------------
    financial = np.mean([
        normalize_positive(data['revenue_growth'], 40, -10),       # %-growth
        normalize_positive(data['profit_margin'], 30, 0),
        normalize_positive(data['roe'], 30, 0),
        normalize_negative(data['debt_to_equity'], 3, 0)
    ])

    # ------------- VALUATION (15%) -------------
    valuation = np.mean([
        normalize_negative(data['pe_ratio'], 60, 5),
        normalize_negative(data['pb_ratio'], 15, 0.5),
        normalize_negative(data['ev_ebitda'], 30, 5)
    ])

    # ------------- GROWTH (15%) -------------
    growth = np.mean([
        normalize_positive(data['future_eps_growth'], 40, -10),
        normalize_positive(data['target_upside'], 50, -20),
        normalize_positive(data['sector_growth'], 20, 0)
    ])

    # ------------- RISK (15%) -------------
    risk = np.mean([
        normalize_negative(data['beta'], 2, 0.6),
        normalize_negative(data['volatility'], 50, 10),
        normalize_positive(data['altman_z'], 4, 1.0),
        normalize_positive(data['interest_coverage'], 20, 0)
    ])

    # ------------- MOMENTUM (10%) -------------
    momentum = np.mean([
        normalize_positive(data['rsi'], 70, 30),  # Mid-range best
        normalize_positive(data['macd_signal'], 3, -3),
        normalize_positive(data['price_above_ma'], 15, -15),
        normalize_positive(data['recent_returns'], 20, -20)
    ])

    # ------------- LIQUIDITY (10%) -------------
    liquidity = np.mean([
        normalize_positive(data['volume_ratio'], 3, 0.2),
        normalize_positive(data['market_cap'], 200000, 1000)
    ])

    # ------------- MANAGEMENT (8%) -------------
    management = np.mean([
        normalize_positive(data['promoter_holding'], 80, 20),
        normalize_negative(data['promoter_pledge'], 50, 0),
        normalize_positive(data['institutional_holding'], 80, 10),
        normalize_negative(data['governance_redflags'], 5, 0)
    ])

    # ------------- INDUSTRY POSITION (7%) -------------
    industry = np.mean([
        normalize_positive(data['industry_outlook'], 10, 1),
        normalize_positive(data['moat_strength'], 10, 1),
        normalize_negative(data['regulation_risk'], 10, 1)
    ])

    # ------------------------------------------------------------
    # 3) WEIGHTED COMPOSITE SCORE
    # ------------------------------------------------------------
    final_score = (
        financial * 0.20 +
        valuation * 0.15 +
        growth * 0.15 +
        risk * 0.15 +
        momentum * 0.10 +
        liquidity * 0.10 +
        management * 0.08 +
        industry * 0.07
    )

    final_score = round(final_score, 2)

    # ------------------------------------------------------------
    # 4) INVESTMENT RECOMMENDATION
    # ------------------------------------------------------------
    if final_score >= 80:
        recommendation = "Strong Buy"
    elif final_score >= 65:
        recommendation = "Buy"
    elif final_score >= 50:
        recommendation = "Hold"
    elif final_score >= 35:
        recommendation = "Sell"
    else:
        recommendation = "Strong Sell"

    # ------------------------------------------------------------
    # 5) RETURN CATEGORY BREAKDOWN
    # ------------------------------------------------------------
    breakdown = {
        "financial": round(financial, 2),
        "valuation": round(valuation, 2),
        "growth": round(growth, 2),
        "risk": round(risk, 2),
        "momentum": round(momentum, 2),
        "liquidity": round(liquidity, 2),
        "management": round(management, 2),
        "industry": round(industry, 2)
    }

    return final_score, recommendation, breakdown

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="Stock Price Prediction (LSTM) By Syed Afroz Ali", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        color: #00B4D8;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #03045E;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #0077B6;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 8px 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #023E8A;
        color: #FFD60A;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📈 AI-Powered Stock Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Using LSTM Neural Networks + Technical Indicators + Advanced Scoring</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------
SECTORS = {
    "Information Technology (IT) & Services": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"
    ],
    "Banking & Financial Services": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS"
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

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Stock_market_icon.png", width=180)
    st.header("🎛️ Prediction Settings")
    sector = st.selectbox("Select Sector", list(SECTORS.keys()))
    ticker = st.selectbox("Select Stock", SECTORS[sector])

    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=1825))
    end_date = st.date_input("End Date", datetime.now())
    epochs = st.slider("Epochs", 10, 200, 50)
    lookback = st.slider("Lookback Days", 30, 120, 60)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_data(ticker, start, end):
    import time
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)

            if data.empty:
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Attempt {attempt + 1}/{max_retries} - Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise ValueError(f"No data retrieved for {ticker}.")

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.dropna(inplace=True)

            if len(data) < 30:
                raise ValueError(f"Insufficient data for {ticker}. Only {len(data)} days available.")

            close_series = data['Close'].squeeze()
            high_series = data['High'].squeeze()
            low_series = data['Low'].squeeze()
            volume_series = data['Volume'].squeeze()

            data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
            data['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
            data['MACD'] = ta.trend.macd_diff(close_series)
            data['RSI'] = ta.momentum.rsi(close_series)
            data['BB_high'] = ta.volatility.bollinger_hband(close_series)
            data['BB_low'] = ta.volatility.bollinger_lband(close_series)
            data['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series)
            data['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
            data['STOCH_K'] = ta.momentum.stoch(high_series, low_series, close_series)
            data['STOCH_D'] = ta.momentum.stoch_signal(high_series, low_series, close_series)
            data['ADX'] = ta.trend.adx(high_series, low_series, close_series)
            data['WilliamsR'] = ta.momentum.williams_r(high_series, low_series, close_series)

            return data

        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"⏳ Attempt {attempt + 1}/{max_retries} failed - Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                st.error(f"❌ Error loading data: {str(e)}")
                raise

try:
    with st.spinner(f'Loading data for {ticker}...'):
        data = load_data(ticker, start_date, end_date)
    st.success(f"✅ Successfully loaded {len(data)} days of data for {ticker}")
except Exception as e:
    st.error(f"❌ Failed to load data. Please try again later.")
    st.stop()

# ------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------
st.markdown(f"### 📊 {ticker} Stock Data Preview")
st.dataframe(data.tail())

st.markdown("#### Interactive Close Price Chart")
st.line_chart(data['Close'])

# ------------------------------------------------------------
# INDICATOR VISUALIZATION
# ------------------------------------------------------------
st.markdown("### 📈 Technical Indicator Charts")

fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
axs[0].plot(data['Close'], label='Close Price', color='black')
axs[0].plot(data['SMA_20'], label='SMA 20', linestyle='--')
axs[0].plot(data['EMA_20'], label='EMA 20', linestyle=':')
axs[0].fill_between(data.index, data['BB_low'], data['BB_high'], color='gray', alpha=0.1)
axs[0].legend(); axs[0].set_title('Price + SMA/EMA + Bollinger Bands')
axs[1].plot(data['MACD'], label='MACD', color='green'); axs[1].set_title('MACD')
axs[2].plot(data['RSI'], label='RSI', color='blue'); axs[2].axhline(70, color='red', linestyle='--'); axs[2].axhline(30, color='green', linestyle='--'); axs[2].set_title('RSI')
axs[3].plot(data['ADX'], label='ADX', color='purple'); axs[3].set_title('ADX')
plt.tight_layout(); st.pyplot(fig)

# ------------------------------------------------------------
# LSTM MODEL
# ------------------------------------------------------------
def prepare_lstm_data(df, feature='Close', lookback=60):
    values = df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y, scaler

if st.button('🚀 Train LSTM Model'):
    with st.spinner('Training model, please wait...'):
        X, y, scaler = prepare_lstm_data(data, 'Close', lookback)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=5)])

        preds = model.predict(X_test)
        preds = scaler.inverse_transform(preds)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.markdown("### 🔮 Model Predictions vs Actual")
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(actual, label='Actual Price', color='#0077B6')
        ax2.plot(preds, label='Predicted Price', color='#FFD60A')
        ax2.set_title(f'{ticker} — LSTM Prediction')
        ax2.legend()
        st.pyplot(fig2)
        st.success('✅ Model training and prediction complete!')

# ------------------------------------------------------------
# FUTURE PREDICTION
# ------------------------------------------------------------
st.markdown("### 🔮 Future Price Prediction")
future_days = st.slider("Predict Next N Days", 5, 30, 7)

if st.button('📊 Generate Future Predictions'):
    with st.spinner('Generating future predictions...'):
        X, y, scaler = prepare_lstm_data(data, 'Close', lookback)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=5)])

        last_sequence = data['Close'].values[-lookback:].reshape(-1, 1)
        last_sequence_scaled = scaler.transform(last_sequence)

        future_predictions = []
        current_sequence = last_sequence_scaled.copy()

        for _ in range(future_days):
            current_sequence_reshaped = current_sequence.reshape(1, lookback, 1)
            next_pred = model.predict(current_sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred).reshape(-1, 1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days, freq='D')

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions.flatten(),
            'Current_Price': data['Close'].iloc[-1],
            'Price_Change': future_predictions.flatten() - data['Close'].iloc[-1],
            'Price_Change_Percent': ((future_predictions.flatten() - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100)
        })

        st.markdown("#### 📈 Future Price Predictions")
        st.dataframe(future_df.style.format({
            'Predicted_Price': '₹{:.2f}',
            'Current_Price': '₹{:.2f}',
            'Price_Change': '₹{:.2f}',
            'Price_Change_Percent': '{:.2f}%'
        }))

        fig3, ax3 = plt.subplots(figsize=(14, 6))
        historical_last = data['Close'].iloc[-30:].values
        ax3.plot(range(len(historical_last)), historical_last, label='Historical (Last 30 Days)', color='#0077B6', linewidth=2)
        ax3.plot(range(len(historical_last)-1, len(historical_last) + future_days),
                [historical_last[-1]] + list(future_predictions.flatten()),
                label='Future Prediction', color='#FFD60A', linewidth=2, linestyle='--', marker='o')
        ax3.set_title(f'{ticker} — Future Price Prediction (Next {future_days} Days)')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Price (₹)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            future_df.to_excel(writer, sheet_name='Future_Predictions', index=False)
            data.tail(50).to_excel(writer, sheet_name='Historical_Data')

        output.seek(0)
        st.download_button(
            label="📥 Download Predictions as Excel",
            data=output,
            file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success('✅ Future predictions generated successfully!')

# ------------------------------------------------------------
# ADVANCED INVESTMENT ANALYSIS
# ------------------------------------------------------------
st.markdown("---")
st.markdown("## 🏢 Advanced Investment Analysis")

try:
    stock_info = yf.Ticker(ticker)
    info = stock_info.info

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Key Metrics")
        st.metric("Current Price", f"₹{info.get('currentPrice', 'N/A')}")
        st.metric("Market Cap", f"₹{info.get('marketCap', 0)/10000000:.2f} Cr" if info.get('marketCap') else "N/A")
        st.metric("52 Week High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.metric("52 Week Low", f"₹{info.get('fiftyTwoWeekLow', 'N/A')}")

    with col2:
        st.markdown("### 💰 Financial Ratios")
        st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A")
        st.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else "N/A")
        st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
        st.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")

    with col3:
        st.markdown("### 📈 Trading Info")
        st.metric("Volume", f"{info.get('volume', 'N/A'):,}" if info.get('volume') else "N/A")
        st.metric("Avg Volume", f"{info.get('averageVolume', 'N/A'):,}" if info.get('averageVolume') else "N/A")
        st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A")
        st.metric("Day Range", f"₹{info.get('dayLow', 'N/A')} - ₹{info.get('dayHigh', 'N/A')}")

    st.markdown("### 📝 Company Overview")
    st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
    st.write(f"**Website:** {info.get('website', 'N/A')}")

    if info.get('longBusinessSummary'):
        with st.expander("📄 Business Summary"):
            st.write(info.get('longBusinessSummary'))

    # ------------------------------------------------------------
    # COMPREHENSIVE SCORING SYSTEM
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 Comprehensive Investment Score")

    # Prepare data for scoring
    current_price = data['Close'].iloc[-1]
    high_52w = info.get('fiftyTwoWeekHigh', current_price)
    low_52w = info.get('fiftyTwoWeekLow', current_price)

    scoring_data = {
        # Financial Performance
        "revenue_growth": info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 10,
        "profit_margin": info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 15,
        "roe": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 12,
        "debt_to_equity": info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0.5,

        # Valuation
        "pe_ratio": info.get('trailingPE', 25) if info.get('trailingPE') else 25,
        "pb_ratio": info.get('priceToBook', 3) if info.get('priceToBook') else 3,
        "ev_ebitda": info.get('enterpriseToEbitda', 15) if info.get('enterpriseToEbitda') else 15,

        # Growth
        "future_eps_growth": info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 10,
        "target_upside": ((info.get('targetMeanPrice', current_price) - current_price) / current_price * 100) if info.get('targetMeanPrice') else 20,
        "sector_growth": 10,  # Default sector growth

        # Risk
        "beta": info.get('beta', 1) if info.get('beta') else 1,
        "volatility": data['Close'].pct_change().std() * 100 * np.sqrt(252),
        "altman_z": 3.0,  # Default Altman Z-score
        "interest_coverage": 8,  # Default interest coverage

        # Momentum
        "rsi": data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else 50,
        "macd_signal": data['MACD'].iloc[-1] if not pd.isna(data['MACD'].iloc[-1]) else 0,
        "price_above_ma": ((current_price - data['SMA_20'].iloc[-1]) / data['SMA_20'].iloc[-1] * 100) if not pd.isna(data['SMA_20'].iloc[-1]) else 0,
        "recent_returns": ((current_price - data['Close'].iloc[-30]) / data['Close'].iloc[-30] * 100) if len(data) >= 30 else 0,

        # Liquidity
        "volume_ratio": info.get('volume', 1) / info.get('averageVolume', 1) if info.get('averageVolume') else 1,
        "market_cap": info.get('marketCap', 0) / 10000000 if info.get('marketCap') else 50000,

        # Management
        "promoter_holding": info.get('heldPercentInsiders', 0.4) * 100 if info.get('heldPercentInsiders') else 40,
        "promoter_pledge": 0,  # Default no pledge
        "institutional_holding": info.get('heldPercentInstitutions', 0.3) * 100 if info.get('heldPercentInstitutions') else 30,
        "governance_redflags": 0,  # Default no red flags

        # Industry
        "industry_outlook": 7,  # Default industry outlook
        "moat_strength": 6,  # Default moat strength
        "regulation_risk": 3,  # Default regulation risk
    }

    final_score, recommendation, breakdown = calculate_investment_score(scoring_data)

    # Display comprehensive score
    score_col1, score_col2 = st.columns([1, 2])

    with score_col1:
        if recommendation == "Strong Buy":
            color = "#00C851"
            emoji = "🟢"
        elif recommendation == "Buy":
            color = "#33B679"
            emoji = "🟢"
        elif recommendation == "Hold":
            color = "#FFBB33"
            emoji = "🟡"
        elif recommendation == "Sell":
            color = "#FF8800"
            emoji = "🟠"
        else:
            color = "#FF4444"
            emoji = "🔴"

        st.markdown(f"### {emoji} Investment Score")
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{final_score}/100</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{recommendation}</h3>", unsafe_allow_html=True)

    with score_col2:
        st.markdown("### 📊 Score Breakdown by Category")
        breakdown_df = pd.DataFrame({
            'Category': ['Financial', 'Valuation', 'Growth', 'Risk', 'Momentum', 'Liquidity', 'Management', 'Industry'],
            'Score': [breakdown['financial'], breakdown['valuation'], breakdown['growth'],
                     breakdown['risk'], breakdown['momentum'], breakdown['liquidity'],
                     breakdown['management'], breakdown['industry']],
            'Weight': ['20%', '15%', '15%', '15%', '10%', '10%', '8%', '7%']
        })
        st.dataframe(breakdown_df.style.format({'Score': '{:.2f}'}), use_container_width=True)

    # Visualization of scores
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    categories = list(breakdown.keys())
    scores = [breakdown[cat] for cat in categories]
    colors_bar = ['#00B4D8' if s >= 60 else '#FFD60A' if s >= 40 else '#FF6B6B' for s in scores]
    ax4.barh(categories, scores, color=colors_bar)
    ax4.set_xlabel('Score (0-100)')
    ax4.set_title('Investment Score Breakdown')
    ax4.set_xlim(0, 100)
    for i, v in enumerate(scores):
        ax4.text(v + 1, i, f'{v:.1f}', va='center')
    plt.tight_layout()
    st.pyplot(fig4)

    st.warning("⚠️ **Disclaimer:** This comprehensive analysis combines technical indicators, financial metrics, and quantitative scoring. It should not be considered as financial advice. Please consult with a certified financial advisor before making investment decisions.")

    # ------------------------------------------------------------
    # DETAILED ANALYSIS FACTORS
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Detailed Analysis Factors")

    with st.expander("🔍 View Detailed Factor Analysis"):
        st.markdown("#### Financial Performance (20% weight)")
        st.write(f"- **Revenue Growth:** {scoring_data['revenue_growth']:.2f}%")
        st.write(f"- **Profit Margin:** {scoring_data['profit_margin']:.2f}%")
        st.write(f"- **Return on Equity (ROE):** {scoring_data['roe']:.2f}%")
        st.write(f"- **Debt to Equity:** {scoring_data['debt_to_equity']:.2f}")
        st.write(f"**Category Score:** {breakdown['financial']:.2f}/100")

        st.markdown("#### Valuation (15% weight)")
        st.write(f"- **P/E Ratio:** {scoring_data['pe_ratio']:.2f}")
        st.write(f"- **P/B Ratio:** {scoring_data['pb_ratio']:.2f}")
        st.write(f"- **EV/EBITDA:** {scoring_data['ev_ebitda']:.2f}")
        st.write(f"**Category Score:** {breakdown['valuation']:.2f}/100")

        st.markdown("#### Growth Potential (15% weight)")
        st.write(f"- **Future EPS Growth:** {scoring_data['future_eps_growth']:.2f}%")
        st.write(f"- **Target Price Upside:** {scoring_data['target_upside']:.2f}%")
        st.write(f"- **Sector Growth Rate:** {scoring_data['sector_growth']:.2f}%")
        st.write(f"**Category Score:** {breakdown['growth']:.2f}/100")

        st.markdown("#### Risk Assessment (15% weight)")
        st.write(f"- **Beta (Volatility vs Market):** {scoring_data['beta']:.2f}")
        st.write(f"- **Price Volatility:** {scoring_data['volatility']:.2f}%")
        st.write(f"- **Altman Z-Score:** {scoring_data['altman_z']:.2f}")
        st.write(f"- **Interest Coverage:** {scoring_data['interest_coverage']:.2f}x")
        st.write(f"**Category Score:** {breakdown['risk']:.2f}/100")

        st.markdown("#### Momentum (10% weight)")
        st.write(f"- **RSI (Relative Strength Index):** {scoring_data['rsi']:.2f}")
        st.write(f"- **MACD Signal:** {scoring_data['macd_signal']:.2f}")
        st.write(f"- **Price vs 20-day MA:** {scoring_data['price_above_ma']:.2f}%")
        st.write(f"- **30-Day Returns:** {scoring_data['recent_returns']:.2f}%")
        st.write(f"**Category Score:** {breakdown['momentum']:.2f}/100")

        st.markdown("#### Liquidity (10% weight)")
        st.write(f"- **Volume Ratio (vs Avg):** {scoring_data['volume_ratio']:.2f}x")
        st.write(f"- **Market Cap:** ₹{scoring_data['market_cap']:.2f} Crores")
        st.write(f"**Category Score:** {breakdown['liquidity']:.2f}/100")

        st.markdown("#### Management Quality (8% weight)")
        st.write(f"- **Promoter Holding:** {scoring_data['promoter_holding']:.2f}%")
        st.write(f"- **Promoter Pledge:** {scoring_data['promoter_pledge']:.2f}%")
        st.write(f"- **Institutional Holding:** {scoring_data['institutional_holding']:.2f}%")
        st.write(f"- **Governance Red Flags:** {scoring_data['governance_redflags']}")
        st.write(f"**Category Score:** {breakdown['management']:.2f}/100")

        st.markdown("#### Industry Position (7% weight)")
        st.write(f"- **Industry Outlook:** {scoring_data['industry_outlook']}/10")
        st.write(f"- **Competitive Moat Strength:** {scoring_data['moat_strength']}/10")
        st.write(f"- **Regulatory Risk:** {scoring_data['regulation_risk']}/10")
        st.write(f"**Category Score:** {breakdown['industry']:.2f}/100")

    # ------------------------------------------------------------
    # QUICK TECHNICAL ANALYSIS
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚡ Quick Technical Analysis Summary")

    quick_col1, quick_col2, quick_col3 = st.columns(3)

    with quick_col1:
        current_rsi = data['RSI'].iloc[-1]
        if current_rsi < 30:
            rsi_status = "🟢 Oversold - Potential Buy"
            rsi_color = "green"
        elif current_rsi > 70:
            rsi_status = "🔴 Overbought - Potential Sell"
            rsi_color = "red"
        else:
            rsi_status = "🟡 Neutral"
            rsi_color = "orange"
        st.markdown(f"**RSI ({current_rsi:.2f}):**")
        st.markdown(f"<span style='color:{rsi_color};'>{rsi_status}</span>", unsafe_allow_html=True)

    with quick_col2:
        if data['MACD'].iloc[-1] > 0:
            macd_status = "🟢 Bullish Momentum"
            macd_color = "green"
        else:
            macd_status = "🔴 Bearish Momentum"
            macd_color = "red"
        st.markdown(f"**MACD:**")
        st.markdown(f"<span style='color:{macd_color};'>{macd_status}</span>", unsafe_allow_html=True)

    with quick_col3:
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        if current_price > sma_20:
            trend_status = "🟢 Uptrend"
            trend_color = "green"
        else:
            trend_status = "🔴 Downtrend"
            trend_color = "red"
        st.markdown(f"**Price vs SMA-20:**")
        st.markdown(f"<span style='color:{trend_color};'>{trend_status}</span>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # RISK WARNINGS
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚠️ Risk Factors to Consider")

    risk_warnings = []

    # High PE ratio warning
    if scoring_data['pe_ratio'] > 40:
        risk_warnings.append(f"❗ **High P/E Ratio ({scoring_data['pe_ratio']:.2f}):** Stock may be overvalued relative to earnings.")

    # High volatility warning
    if scoring_data['volatility'] > 40:
        risk_warnings.append(f"❗ **High Volatility ({scoring_data['volatility']:.2f}%):** Stock shows significant price fluctuations.")

    # High debt warning
    if scoring_data['debt_to_equity'] > 2:
        risk_warnings.append(f"❗ **High Debt-to-Equity ({scoring_data['debt_to_equity']:.2f}):** Company has high leverage.")

    # Overbought warning
    if current_rsi > 75:
        risk_warnings.append(f"❗ **Overbought Condition (RSI: {current_rsi:.2f}):** Stock may be due for a correction.")

    # Low liquidity warning
    if scoring_data['volume_ratio'] < 0.5:
        risk_warnings.append(f"❗ **Low Trading Volume:** Current volume is significantly below average.")

    if risk_warnings:
        for warning in risk_warnings:
            st.warning(warning)
    else:
        st.success("✅ No major risk factors detected in current analysis.")

    # ------------------------------------------------------------
    # ACTIONABLE INSIGHTS
    # ------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💡 Actionable Insights")

    insights = []

    # Price momentum insight
    recent_change = ((current_price - data['Close'].iloc[-30]) / data['Close'].iloc[-30] * 100) if len(data) >= 30 else 0
    if recent_change > 10:
        insights.append(f"📈 Stock has gained {recent_change:.2f}% in the last 30 days - Strong upward momentum")
    elif recent_change < -10:
        insights.append(f"📉 Stock has declined {abs(recent_change):.2f}% in the last 30 days - Consider waiting for stabilization")

    # Volume insight
    if scoring_data['volume_ratio'] > 1.5:
        insights.append("📊 Trading volume is significantly above average - Increased market interest")

    # Valuation insight
    if scoring_data['target_upside'] > 20:
        insights.append(f"🎯 Analyst target suggests {scoring_data['target_upside']:.2f}% upside potential")
    elif scoring_data['target_upside'] < -10:
        insights.append(f"⚠️ Analyst target suggests {abs(scoring_data['target_upside']):.2f}% downside risk")

    # Technical setup insight
    if current_rsi < 40 and data['MACD'].iloc[-1] > 0:
        insights.append("✨ Favorable technical setup - RSI recovering with positive MACD")

    # Display insights
    if insights:
        for insight in insights:
            st.info(insight)

except Exception as e:
    st.error(f"Could not complete advanced analysis: {str(e)}")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>📈 AI-Powered Stock Price Prediction App</strong></p>
        <p>Built with ❤️ by Syed Afroz Ali | Powered by LSTM Neural Networks</p>
        <p style='font-size: 12px;'>Data provided by Yahoo Finance | Technical Analysis powered by TA-Lib</p>
        <p style='font-size: 12px;'>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)
