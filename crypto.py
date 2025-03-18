import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("Cryptocurrency Trend Prediction")

crypto_symbol = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD):", "BTC-USD")
days = st.slider("Select number of past days for training:", 30, 365, 60)
pred_days = st.slider("Select number of days to predict:", 1, 30, 7)

if st.button("Train and Predict"):
    df = yf.download(crypto_symbol, period=f"{days+pred_days}d", interval="1d")
    df = df[['Close']]
    st.write("### Raw Data")
    st.write(df.tail())
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df.values)
    
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 10
    X_train, y_train = create_sequences(scaled_data[:-pred_days], seq_length)
    X_test, y_test = create_sequences(scaled_data, seq_length)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1)
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = df['Close'].values[-len(predictions):]
    
    st.write("### Predictions vs Actual Prices")
    fig, ax = plt.subplots()
    ax.plot(actual_prices, label='Actual Price')
    ax.plot(predictions, label='Predicted Price')
    ax.legend()
    st.pyplot(fig)
    
    st.success("Prediction completed!")