# 📈 AI Stock Price Prediction using LSTM

## 📌 Overview

The **AI Stock Price Prediction** project is a machine learning application that predicts future stock prices of leading Indian companies using a **Long Short-Term Memory (LSTM)** neural network. The application fetches historical stock market data from **Yahoo Finance**, trains an LSTM model, and forecasts the next **30 business days** of stock prices through an interactive **Streamlit** dashboard.

This project demonstrates the practical application of **Deep Learning**, **Time Series Forecasting**, and **Data Visualization** using Python.

---

# ✨ Features

* 📊 Real-time stock data from Yahoo Finance
* 🤖 LSTM-based Deep Learning model
* 📈 Predicts the next 30 business days
* 🏢 Supports multiple Indian companies
* 📅 Monthly stock performance visualization
* 📉 Actual vs Predicted price comparison
* ⚡ Interactive Streamlit dashboard
* 📂 Automatic data preprocessing and scaling
* 🔄 Cached model for faster execution

---

# 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* TensorFlow
* Keras
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* yFinance

---

# 🧠 Machine Learning Workflow

```text
Historical Stock Data
          │
          ▼
 Data Preprocessing
          │
          ▼
 Normalization (MinMaxScaler)
          │
          ▼
 Sequence Generation
          │
          ▼
 LSTM Model Training
          │
          ▼
 Stock Price Prediction
          │
          ▼
30-Day Future Forecast
```

---

# 🏢 Supported Companies

* Reliance Industries
* Tata Consultancy Services (TCS)
* Infosys
* HDFC Bank
* ICICI Bank
* State Bank of India
* Bharti Airtel
* Hindustan Unilever
* ITC Limited
* Bajaj Finance
* Wipro
* Larsen & Toubro

---

# 📊 Dashboard Features

### Stock Selection

Choose any supported Indian company from the sidebar.

### Actual vs Predicted Prices

Visual comparison between real stock prices and LSTM predictions.

### 30-Day Forecast

Forecasts the next 30 business days using the trained LSTM model.

### Monthly Performance

Visualizes stock performance for a selected month in the latest available year.

---

# 🚀 How to Run

## Clone Repository

```bash
git clone https://github.com/Satheessk/Stock-Price-Prediction.git
```

## Navigate to Project

```bash
cd Stock-Price-Prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```text
Stock-Price-Prediction/

│── app.py
│── requirements.txt
│── README.md
│── screenshots/
```

---

# 📈 Machine Learning Concepts Used

* Long Short-Term Memory (LSTM)
* Time Series Forecasting
* Sequential Neural Networks
* MinMax Scaling
* Sliding Window Technique
* Deep Learning
* Data Visualization

---

# 📊 Model Details

| Parameter       | Value              |
| --------------- | ------------------ |
| Model           | LSTM               |
| Optimizer       | Adam               |
| Loss Function   | Mean Squared Error |
| Sequence Length | 60 Days            |
| Batch Size      | 32                 |
| Epochs          | 8                  |

---

# 📸 Screenshots

> Screenshots will be added soon.

* Dashboard
* Actual vs Predicted Graph
* 30-Day Forecast
* Monthly Performance

---

# 🔮 Future Enhancements

* Model accuracy metrics (RMSE, MAE, R²)
* Interactive Plotly charts
* Multiple prediction intervals
* Candlestick chart visualization
* Buy/Sell trend indicators
* Export predictions as CSV
* News sentiment analysis
* Portfolio performance dashboard

---

# 🎯 Learning Outcomes

This project helped in understanding:

* Deep Learning fundamentals
* LSTM Neural Networks
* Time Series Forecasting
* Data Preprocessing
* Streamlit Application Development
* Data Visualization
* Machine Learning Model Deployment

---

# ⚠️ Disclaimer

This project is developed for **educational and research purposes only**. Stock market predictions are based on historical data and machine learning models and should **not** be considered financial or investment advice.

---

# 👨‍💻 Author

**Satheeskumar G**

* GitHub: https://github.com/Satheessk

---

# ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.
