# Cryptocurrency Trend Prediction

This project is a **Cryptocurrency Trend Prediction** tool built with **Streamlit, TensorFlow, and yFinance**. It allows users to train an LSTM model on historical cryptocurrency price data and make short-term predictions.

## Features
- Fetch real-time cryptocurrency data using **yFinance**
- Normalize and preprocess data with **MinMaxScaler**
- Train an **LSTM (Long Short-Term Memory)** neural network for trend prediction
- Visualize actual vs predicted prices using **Matplotlib**
- Interactive UI using **Streamlit**

## Installation
### Prerequisites
Ensure you have Python installed along with the following dependencies:

```sh
pip install streamlit numpy pandas yfinance tensorflow scikit-learn matplotlib
```

## Usage
Run the Streamlit app with:

```sh
streamlit run app.py
```

### How It Works
1. Enter the cryptocurrency symbol (e.g., BTC-USD).
2. Select the number of past days for training and prediction horizon.
3. Click the **Train and Predict** button.
4. View the actual vs predicted prices plotted on a graph.

## Project Structure
```
📂 Cryptocurrency-Trend-Prediction
│── app.py          # Main Streamlit application
│── requirements.txt # List of dependencies
│── README.md       # Documentation
```

## Model Details
- Uses **LSTM layers** with dropout for time series forecasting.
- Optimized with the **Adam** optimizer and mean squared error loss.
- The sequence length for training is set to **10 days**.
- Trains on **16 batch size** for **20 epochs**.
