import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Cryptocurrency Trend Prediction")

# User inputs
crypto_symbol = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD):", "BTC-USD")
days = st.slider("Select number of past days for training:", 30, 365, 60)
pred_days = st.slider("Select number of days to predict:", 1, 30, 7)

if st.button("Train and Predict"):
  # Download historical data
  df = yf.download(crypto_symbol, period=f"{days+pred_days}d", interval="1d")
  print(df.index)
  df = df[['Open', 'High', 'Low', 'Close']]
  st.write("### Raw Data")
  st.write(df.tail())

  # Scale the data
  scaler = MinMaxScaler(feature_range=(0, 1))
  # scaled_data = scaler.fit_transform(df.values)
  scaled_data = scaler.fit_transform(df[['Close']].values)

  # Function to create sequences for LSTM
  def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
      X.append(data[i:i+seq_length])
      y.append(data[i+seq_length])
    return np.array(X), np.array(y)

  seq_length = 10 # Sequence length
  X_train, y_train = create_sequences(scaled_data[:-pred_days], seq_length)
  X_test, y_test = create_sequences(scaled_data, seq_length)

  # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  # Build the LSTM model
  model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
  ])
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model.fit(X_train, y_train, batch_size=16, epochs=40, verbose=1)

  # Predict known data
  predictions = model.predict(X_test)
  predictions = scaler.inverse_transform(predictions)
  actual_prices = df['Close'].values[-len(predictions):]

  # Generate future predictions
  future_predictions = []
  last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1) # Last known sequence

  for _ in range(pred_days):
    next_prediction = model.predict(last_sequence) # Output shape (1,1)
    
    # Reshape next_prediction to match sequence shape
    next_prediction = np.reshape(next_prediction, (1, 1, 1)) # Shape (1,1,1)
    
    # Update sequence: Remove first value, append new prediction
    last_sequence = np.concatenate((last_sequence[:, 1:, :], next_prediction), axis=1)

    # Store the predicted value
    future_predictions.append(next_prediction[0, 0, 0]) 

  future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

  st.write("### Candlestick Chart")

  fig = go.Figure([go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Candlestick",
    increasing_line_color='lime',
    decreasing_line_color='red',
    increasing_fillcolor='rgba(0,255,0,0.5)',
    decreasing_fillcolor='rgba(255,0,0,0.5)'
  )])

  # Optional: Add prediction as line chart
  fig.add_trace(go.Scatter(
    x=df.index[-len(predictions):],
    y=predictions.flatten(),
    mode='lines',
    name='Predicted (Historical)',
    line=dict(color='orange', width = 1.5)
  ))

  # Future predictions
  future_dates = pd.date_range(start=df.index[-1], periods=pred_days+1)[1:]
  fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_predictions.flatten(),
    mode='lines',
    name='Future Prediction',
    line=dict(dash='dash', color='red', width = 1.5)
  ))

  fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
  st.plotly_chart(fig)

  # # Plot results
  # st.write("### Predictions vs Actual Prices")
  # fig, ax = plt.subplots()
  # ax.plot(actual_prices, label="Actual Price", color='blue')
  # ax.plot(predictions, label="Predicted Price (Historical)", color='orange')

  # # Plot future predictions
  # future_dates = pd.date_range(start=df.index[-1], periods=pred_days+1)[1:]
  # ax.plot(range(len(actual_prices), len(actual_prices) + pred_days), future_predictions, label="Future Prediction", linestyle='dashed', color='red')

  # ax.legend()
  # st.pyplot(fig)

  st.success("Prediction completed!")