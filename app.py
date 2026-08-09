import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Page Configuration ---
st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")

# --- Sidebar UI ---
st.sidebar.title("📈 Stock Price Predictor (INR)")
st.sidebar.write("Select a stock to train the LSTM model and view performance metrics.")

companies = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC Limited": "ITC.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Wipro": "WIPRO.NS",
    "Larsen & Toubro": "LT.NS"
}

selected_company = st.sidebar.selectbox("Select Company", list(companies.keys()))
ticker = companies[selected_company]

# --- Month Selection ---
month_options = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
selected_month_num = st.sidebar.selectbox(
    "Select Month for Historical View",
    list(month_options.keys()),
    format_func=lambda x: month_options[x]
)

# --- Fetch Data ---
start_date = "2015-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

st.title(f"🏢 {selected_company} ({ticker})")
st.write("Fetching historical market data...")

data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if data.empty:
    st.error("No market data found for this ticker.")
    st.stop()

# Clean MultiIndex columns if returned by yfinance
if isinstance(data.columns, pd.MultiIndex):
    data = data.xs(ticker, level=1, axis=1)

st.success(f"✅ Data loaded successfully ({len(data)} trading days).")

# --- Cached Model Training (Direct Multi-Step LSTM) ---
@st.cache_resource
def train_direct_lstm_model(close_series):
    prices = close_series.values.reshape(-1, 1)
    
    # 1. Train/Test Split BEFORE scaling to prevent Data Leakage
    train_size = int(len(prices) * 0.8)
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]

    # 2. Fit Scaler ONLY on Training Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_prices)
    scaled_test = scaler.transform(test_prices)
    
    scaled_full = np.vstack((scaled_train, scaled_test))

    seq_length = 60
    forecast_length = 30
    
    # Create sequence windows
    def create_direct_sequences(dataset, start_idx, end_idx):
        x, y = [], []
        for i in range(start_idx + seq_length, end_idx - forecast_length + 1):
            x.append(dataset[i - seq_length:i, 0])
            y.append(dataset[i:i + forecast_length, 0])
        return np.array(x), np.array(y)

    x_train, y_train = create_direct_sequences(scaled_full, 0, train_size)
    x_test, y_test = create_direct_sequences(scaled_full, train_size - seq_length, len(scaled_full))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # --- Build Architecture ---
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(forecast_length)  # Direct output layer for 30 business days
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=0)

    return model, scaler, scaled_full, x_test, y_test

close_series = data['Close'].squeeze()
model, scaler, scaled_data, x_test, y_test = train_direct_lstm_model(close_series)
st.info("💡 LSTM Model trained & cached.")

# --- Evaluate Predictions ---
test_preds = model.predict(x_test, verbose=0)

# Unscale 1-day ahead target vs predictions for evaluation
pred_1day = scaler.inverse_transform(test_preds[:, 0].reshape(-1, 1))
real_1day = scaler.inverse_transform(y_test[:, 0].reshape(-1, 1))

# --- Calculate Model Performance Metrics ---
mae = mean_absolute_error(real_1day, pred_1day)
rmse = np.sqrt(mean_squared_error(real_1day, pred_1day))
mape = np.mean(np.abs((real_1day - pred_1day) / real_1day)) * 100
r2 = r2_score(real_1day, pred_1day)

# --- Display Evaluation Metrics ---
st.subheader("📏 Model Evaluation Metrics (Test Set)")
m1, m2, m3, m4 = st.columns(4)

m1.metric("Mean Absolute Error (MAE)", f"₹{mae:.2f}")
m2.metric("Root Mean Squared Error (RMSE)", f"₹{rmse:.2f}")
m3.metric("Mean Absolute % Error (MAPE)", f"{mape:.2f}%")
m4.metric("R² Score", f"{r2:.3f}")

st.markdown("---")

# --- Visualizing 30 Business Days Forecast ---
st.subheader("🔮 Next 30 Business Days Forecast")

last_60_days = scaled_data[-60:].reshape(1, 60, 1)
future_preds_scaled = model.predict(last_60_days, verbose=0)
future_predictions = scaler.inverse_transform(future_preds_scaled).reshape(-1, 1)

future_dates = pd.bdate_range(data.index[-1] + timedelta(days=1), periods=30)

fig_future, ax_future = plt.subplots(figsize=(10, 4))
ax_future.plot(future_dates, future_predictions, marker='o', color='#2ca02c', label='Predicted Price (₹)')
ax_future.set_xlabel("Future Dates")
ax_future.set_ylabel("Price (₹)")
ax_future.set_title(f"{selected_company} - Direct 30-Day Forecast")
ax_future.legend()
ax_future.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig_future)

# --- Visualizing Actual vs Predicted Prices ---
st.subheader("📊 Test Set Evaluation: Actual vs Predicted Prices")
fig_test, ax_test = plt.subplots(figsize=(10, 4))
test_dates = data.index[-len(real_1day):]
ax_test.plot(test_dates, real_1day, color='#1f77b4', label='Actual Price (₹)')
ax_test.plot(test_dates, pred_1day, color='#d62728', linestyle='--', label='1-Day Ahead Prediction (₹)')
ax_test.set_xlabel("Date")
ax_test.set_ylabel("Price (₹)")
ax_test.legend()
ax_test.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig_test)

# --- Historical Monthly Performance ---
latest_year = data.index.max().year
st.subheader(f"📅 Historical Context: {month_options[selected_month_num]} {latest_year}")

monthly_data = data[
    (data.index.year == latest_year) & 
    (data.index.month == selected_month_num)
]

if monthly_data.empty:
    st.warning(f"No historical data recorded for {month_options[selected_month_num]} {latest_year}.")
else:
    fig_month, ax_month = plt.subplots(figsize=(10, 4))
    ax_month.plot(monthly_data.index, monthly_data['Close'].squeeze(), color='#9467bd', marker='o', linestyle='-')
    ax_month.set_xlabel("Date")
    ax_month.set_ylabel("Closing Price (₹)")
    ax_month.set_title(f"{selected_company} - {month_options[selected_month_num]} {latest_year}")
    ax_month.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_month)

st.caption("⚠️ Disclaimer: Educational use only. Stock market trading carries financial risk.")