import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from keras_tuner.tuners import RandomSearch

st.title("Cryptocurrency Trend Prediction")

# User inputs
crypto_symbol = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD):", "BTC-USD")
days = st.slider("Select number of past days for training:", 60, 730, 365)
pred_days = st.slider("Select number of days to predict:", 7, 365, 30)
seq_length = st.slider("Select sequence length for LSTM:", 5, 60, 20)
use_bidirectional = st.checkbox("Use Bidirectional LSTM", False)
scaler_type = st.selectbox("Select Scaling Method:", ["MinMaxScaler", "StandardScaler"])
optimize_model = st.checkbox("Optimize Model (Hyperparameter Tuning - may take time)", False)
n_trials = st.slider("Number of Tuning Trials (if optimizing):", 5, 20, 10)
epochs = st.slider("Number of Training Epochs:", 20, 100, 50)
batch_size = st.selectbox("Batch Size:", [16, 32, 64, 128])

@st.cache_data(ttl=3600)
def load_data(symbol, period):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d")
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(hp, input_shape, bidirectional=False):
    model = Sequential()
    hp_units_lstm1 = hp.Int('units_lstm1', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=hp_units_lstm1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.3, step=0.1)))
    hp_num_layers = hp.Int('num_lstm_layers', min_value=1, max_value=2)
    for i in range(hp_num_layers -1):
        hp_units_lstm_n = hp.Int(f'units_lstm_{i+2}', min_value=32, max_value=128, step=32)
        model.add(LSTM(units=hp_units_lstm_n, return_sequences=True))
        model.add(Dropout(hp.Float(f'dropout_{i+2}', min_value=0.1, max_value=0.3, step=0.1)))
    hp_units_lstm_final = hp.Int('units_lstm_final', min_value=32, max_value=64, step=32)
    model.add(LSTM(units=hp_units_lstm_final, return_sequences=False))
    model.add(Dense(hp.Int('units_dense1', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(1))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mean_squared_error')
    return model

def build_simple_lstm_model(input_shape, bidirectional=False):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)) if bidirectional else LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)) if bidirectional else LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if st.button("Train and Predict"):
    df = load_data(crypto_symbol, f"{days+pred_days}d")
    st.write("### Raw Data")
    st.write(df.tail())

    # Prepare data
    data = df['Close'].values.reshape(-1, 1)

    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, seq_length)
    X_train = X[:-pred_days]
    y_train = y[:-pred_days]
    X_test = X[-pred_days:]
    y_test = y[-pred_days:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    st.write("### Training the Model...")
    if optimize_model:
        tuner = RandomSearch(
            lambda hp: build_lstm_model(hp, input_shape=(seq_length, 1), bidirectional=use_bidirectional),
            objective='val_loss',
            max_trials=n_trials,
            directory='crypto_tuner',
            project_name='lstm_optimization'
        )

        # Split training data for validation
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        tuner.search(X_train_tune, y_train_tune,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(X_val_tune, y_val_tune),
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = build_lstm_model(best_hps, input_shape=(seq_length, 1), bidirectional=use_bidirectional)
        st.write(f"Best Hyperparameters found: {best_hps.values}")

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    else:
        model = build_simple_lstm_model(input_shape=(seq_length, 1), bidirectional=use_bidirectional)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict known data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = data[-len(predictions):]

    # Generate future predictions
    future_predictions = []
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)

    for _ in range(pred_days):
        next_prediction = model.predict(last_sequence)
        next_prediction = np.reshape(next_prediction, (1, 1, 1))
        last_sequence = np.concatenate((last_sequence[:, 1:, :], next_prediction), axis=1)
        future_predictions.append(next_prediction[0, 0, 0])

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    st.write("### Price Prediction")
    df_predictions = pd.DataFrame(index=df.index[-len(predictions):], data={'Actual': actual_prices.flatten(), 'Predicted': predictions.flatten()})
    st.write(df_predictions)

    st.write("### Candlestick Chart with Predictions")
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

    fig.add_trace(go.Scatter(
        x=df.index[-len(predictions):],
        y=predictions.flatten(),
        mode='lines',
        name='Predicted (Historical)',
        line=dict(color='orange', width=1.5)
    ))

    future_dates = pd.date_range(start=df.index[-1], periods=pred_days + 1)[1:]
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions.flatten(),
        mode='lines',
        name='Future Prediction',
        line=dict(dash='dash', color='red', width=1.5)
    ))

    fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

    # Evaluate the model
    rmse = root_mean_squared_error(actual_prices, predictions)
    mae = mean_absolute_error(actual_prices, predictions)
    r_squared = r2_score(actual_prices, predictions)
    st.write("### Prediction Evaluation")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"R-squared: {r_squared:.4f}")

    st.success("Prediction completed!")