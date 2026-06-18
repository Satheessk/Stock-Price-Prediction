   # 📊 Streamlit LSTM Stock Predictor (Indian Stocks)
# -------------------------------------------------
# Fast version with cached model and better visualization order

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Sidebar UI ---
st.sidebar.title("📈 Stock Price Predictor (INR)")
st.sidebar.write("Choose a company to view prediction")

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

# --- Month Selection for New Visualization ---
month_options = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
selected_month_num = st.sidebar.selectbox(
    "Select Month for Visualization",
    list(month_options.keys()),
    format_func=lambda x: month_options[x]
)

# --- Fetch Data ---
start_date = "2015-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

st.write(f"### 🏢 {selected_company}")
st.write(f"Fetching latest data for `{ticker}`...")
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if data.empty:
    st.error("No data found for this company.")
    st.stop()

st.success(f"✅ Data loaded ({len(data)} rows) — includes today's data if available.")
st.dataframe(data.tail())

# --- Cache preprocessing and model to speed up ---
@st.cache_resource
def train_lstm_model(data):
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    def create_sequences(dataset, seq_length=60):
        x, y = [], []
        for i in range(seq_length, len(dataset)):
            x.append(dataset[i-seq_length:i, 0])
            y.append(dataset[i, 0])
        return np.array(x), np.array(y)

    sequence_length = 60
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]

    x_train, y_train = create_sequences(train_data)
    x_test, y_test = create_sequences(test_data)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # --- Build & train model ---
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=8, batch_size=32, verbose=0)  # smaller epochs = faster

    return model, scaler, scaled_data, x_test, y_test

model, scaler, scaled_data, x_test, y_test = train_lstm_model(data)
st.info("💡 LSTM model trained (cached for reuse).")

# --- Predictions ---
predicted = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- Forecast Next 30 Days FIRST ---
st.subheader("🔮 Next 30 Days Forecast")

sequence_length = 60
last_60_days = scaled_data[-sequence_length:]
future_input = last_60_days.reshape(1, sequence_length, 1)
future_predictions = []

for _ in range(30):
    next_price = model.predict(future_input)[0][0]
    future_predictions.append(next_price)
    new_input = np.append(future_input[:, 1:, :], [[[next_price]]], axis=1)
    future_input = new_input

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.bdate_range(data.index[-1] + timedelta(days=1), periods=30)

fig_future, ax_future = plt.subplots(figsize=(10, 5))
ax_future.plot(future_dates, future_predictions, marker='o', color='green', label='Predicted Price (₹)')
ax_future.set_xlabel("Future Dates")
ax_future.set_ylabel("Predicted Price (₹)")
ax_future.set_title(f"{selected_company} - Next 30 Business Days Forecast")
ax_future.legend()
ax_future.grid(True)
st.pyplot(fig_future)

# --- Actual vs Predicted Prices ---
st.subheader("📊 Actual vs Predicted Prices (Test Set)")
fig, ax = plt.subplots(figsize=(10, 5))
test_dates = data.index[-len(real_prices):]
ax.plot(test_dates, real_prices, color='blue', label='Actual Price (₹)')
ax.plot(test_dates, predicted_prices, color='red', label='Predicted Price (₹)')
ax.set_xlabel("Date")
ax.set_ylabel("Price (₹)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Monthly Visualization ---
st.subheader(f"📅 {month_options[selected_month_num]} Performance (Most Recent Year)")
latest_year = data.index.max().year
monthly_data = data[
    (data.index.year == latest_year) & 
    (data.index.month == selected_month_num)
]

if monthly_data.empty:
    st.warning(f"No data for {month_options[selected_month_num]} {latest_year}.")
else:
    fig_month, ax_month = plt.subplots(figsize=(10, 5))
    ax_month.plot(monthly_data.index, monthly_data['Close'], color='purple', marker='.', linestyle='-')
    ax_month.set_xlabel("Date")
    ax_month.set_ylabel("Closing Price (₹)")
    ax_month.set_title(f"{selected_company} - {month_options[selected_month_num]} {latest_year} Closing Prices")
    ax_month.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_month)

st.caption("⚠️ Disclaimer: Educational use only. Not financial advice.")
