import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras_tuner.tuners import RandomSearch, BayesianOptimization
import math

st.title("Cryptocurrency Trend Prediction")

# User inputs
crypto_symbol = st.text_input("Enter Cryptocurrency Symbol (e.g., BTC-USD):", "BTC-USD")
days = st.slider("Select number of past days for training:", 60, 730, 365)
pred_days = st.slider("Select number of days to predict:", 7, 365, 30)
seq_length = st.slider("Select sequence length for time series models:", 5, 60, 20)

# Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose Model Type:", ["LSTM", "GRU", "Bidirectional LSTM", "Random Forest"])

# Advanced model options
st.sidebar.header("Advanced Model Options")
scaler_type = st.sidebar.selectbox("Scaling Method:", ["MinMaxScaler", "StandardScaler", "RobustScaler"])
features = st.sidebar.multiselect("Features to use:", 
                                ["Close", "Open", "High", "Low", "Volume"], 
                                default=["Close"])
predict_full_history = st.sidebar.checkbox("Predict Full Historical Timeline", True)

# Training options
st.sidebar.header("Training Options")
test_size = st.sidebar.slider("Test Split (% of data):", 0.05, 0.3, 0.2, 0.05)
validation_split = st.sidebar.slider("Validation Split (% of training data):", 0.0, 0.3, 0.2, 0.05)

# Neural network specific options
if model_choice in ["LSTM", "GRU", "Bidirectional LSTM"]:
    optimize_model = st.sidebar.checkbox("Optimize Model (Hyperparameter Tuning)", False)
    tuner_type = st.sidebar.selectbox("Tuner Type:", ["Random Search", "Bayesian Optimization"])
    n_trials = st.sidebar.slider("Number of Tuning Trials (if optimizing):", 5, 30, 10)
    epochs = st.sidebar.slider("Number of Training Epochs:", 20, 200, 50)
    batch_size = st.sidebar.selectbox("Batch Size:", [16, 32, 64, 128])
    early_stopping = st.sidebar.checkbox("Use Early Stopping", True)
    reduce_lr = st.sidebar.checkbox("Use Learning Rate Reduction", True)
    patience = st.sidebar.slider("Patience for Early Stopping:", 5, 20, 10)

# Random Forest specific options
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees:", 50, 500, 100)
    max_depth = st.sidebar.slider("Maximum Depth:", 5, 50, 20)
    min_samples_split = st.sidebar.slider("Minimum Samples Split:", 2, 20, 5)
    min_samples_leaf = st.sidebar.slider("Minimum Samples Leaf:", 1, 10, 2)
    max_features = st.sidebar.selectbox("Max Features:", ["auto", "sqrt", "log2"])
    n_jobs = st.sidebar.slider("Number of Jobs (Parallelism):", -1, 8, -1)

st.sidebar.header("Trading Recommendations")
show_recommendations = st.sidebar.checkbox("Show Trading Recommendations", True)
risk_tolerance = st.sidebar.select_slider(
    "Your Risk Tolerance:",
    options=["Very Low", "Low", "Medium", "High", "Very High"],
    value="Medium"
)

@st.cache_data(ttl=3600)
def load_data(symbol, period):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{period}d", interval="1d")
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Target is always the Close price
    return np.array(X), np.array(y)

def create_rf_dataset(data, seq_length):
    """Create a flattened dataset for Random Forest"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Flatten the sequence into a single feature vector
        X.append(data[i:i+seq_length].flatten())
        y.append(data[i+seq_length, 0])  # Target is always the Close price
    return np.array(X), np.array(y)

def generate_trading_recommendations(future_df, current_price):
    """Generate trading recommendations based on predicted prices"""
    recommendations = []
    
    # Get the predicted prices
    future_prices = future_df['Predicted'].values
    
    # Calculate expected returns for different time horizons
    short_term_return = (future_prices[6] / current_price - 1) * 100  # 1 week
    if len(future_prices) > 7:
        medium_term_return = (future_prices[13] / current_price - 1) * 100  # 2 weeks
    if len(future_prices) > 14:
        long_term_return = (future_prices[-1] / current_price - 1) * 100  # End of prediction period
    else:
        long_term_return = 0
    
    # Generate recommendations based on the expected returns
    recommendations.append({
        'timeframe': 'Short-term (1 week)',
        'expected_return': short_term_return,
        'action': 'Buy' if short_term_return > 5 else 'Sell' if short_term_return < -3 else 'Hold',
        'confidence': min(abs(short_term_return) / 10 * 100, 100),
        'reasoning': f"Expected {'gain' if short_term_return > 0 else 'loss'} of {abs(short_term_return):.2f}% in 1 week"
    })
    if len(future_prices) > 7:
        recommendations.append({
            'timeframe': 'Medium-term (2 weeks)',
            'expected_return': medium_term_return,
            'action': 'Buy' if medium_term_return > 8 else 'Sell' if medium_term_return < -5 else 'Hold',
            'confidence': min(abs(medium_term_return) / 15 * 100, 100),
            'reasoning': f"Expected {'gain' if medium_term_return > 0 else 'loss'} of {abs(medium_term_return):.2f}% in 2 weeks"
        })
    if len(future_prices) > 14:
        recommendations.append({
            'timeframe': 'Long-term (End of forecast)',
            'expected_return': long_term_return,
            'action': 'Buy' if long_term_return > 12 else 'Sell' if long_term_return < -8 else 'Hold',
            'confidence': min(abs(long_term_return) / 20 * 100, 100),
            'reasoning': f"Expected {'gain' if long_term_return > 0 else 'loss'} of {abs(long_term_return):.2f}% by end of forecast period"
        })
    
    # Calculate trend strength and volatility indicators
    price_changes = np.diff(future_prices) / future_prices[:-1] * 100
    trend_strength = np.mean(price_changes)
    volatility = np.std(price_changes)
    
    # Add overall summary recommendation
    if long_term_return != 0:
        if long_term_return > 15 and trend_strength > 0.5:
            overall = "Strong Buy"
            summary = "Strong upward trend detected with significant potential gains."
        elif long_term_return > 8 and trend_strength > 0.2:
            overall = "Buy"
            summary = "Positive trend with good potential for gains."
        elif long_term_return < -12 and trend_strength < -0.5:
            overall = "Strong Sell"
            summary = "Strong downward trend detected with significant potential losses."
        elif long_term_return < -5 and trend_strength < -0.2:
            overall = "Sell"
            summary = "Negative trend with potential for losses."
        elif volatility > 5:
            overall = "Hold/Neutral"
            summary = "High volatility detected. Consider waiting for clearer signals."
        else:
            overall = "Hold/Neutral"
            summary = "No strong trend detected. Market appears to be ranging."
    else:
        overall = "Unknown"
        summary = "Not enough data to recommend the trend."
    
    
    # Calculate risk level based on volatility
    if volatility > 8:
        risk = "High"
    elif volatility > 4:
        risk = "Medium"
    else:
        risk = "Low"
    
    return {
        'detailed_recommendations': recommendations,
        'overall_recommendation': overall,
        'summary': summary,
        'trend_strength': trend_strength,
        'volatility': volatility,
        'risk_level': risk
    }

def build_model_hp(hp, input_shape, model_type):
    model = Sequential()
    
    # First layer
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32)
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    
    if model_type == "LSTM":
        model.add(LSTM(units=hp_units_1, return_sequences=True, input_shape=input_shape))
    elif model_type == "GRU":
        model.add(GRU(units=hp_units_1, return_sequences=True, input_shape=input_shape))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(units=hp_units_1, return_sequences=True, input_shape=input_shape)))
    
    model.add(Dropout(hp_dropout_1))
    
    # Hidden layers
    hp_num_layers = hp.Int('num_layers', min_value=1, max_value=3)
    
    for i in range(hp_num_layers - 1):
        hp_units = hp.Int(f'units_{i+2}', min_value=32, max_value=256, step=32)
        hp_dropout = hp.Float(f'dropout_{i+2}', min_value=0.1, max_value=0.5, step=0.1)
        return_seq = i < hp_num_layers - 2  # Only last RNN layer doesn't return sequences
        
        if model_type == "LSTM":
            model.add(LSTM(units=hp_units, return_sequences=return_seq))
        elif model_type == "GRU":
            model.add(GRU(units=hp_units, return_sequences=return_seq))
        elif model_type == "Bidirectional LSTM":
            model.add(Bidirectional(LSTM(units=hp_units, return_sequences=return_seq)))
        
        model.add(Dropout(hp_dropout))
    
    # Dense layers
    hp_dense_layers = hp.Int('dense_layers', min_value=1, max_value=2)
    for i in range(hp_dense_layers):
        hp_dense_units = hp.Int(f'dense_units_{i+1}', min_value=16, max_value=128, step=16)
        model.add(Dense(hp_dense_units, activation='relu'))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    
    if hp_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
        
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_simple_model(input_shape, model_type="LSTM"):
    model = Sequential()
    
    if model_type == "LSTM":
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
    elif model_type == "GRU":
        model.add(GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(32, return_sequences=False))
    elif model_type == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
    
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if st.button("Train and Predict"):
    # Load more days to ensure we have enough data after filtering
    df = load_data(crypto_symbol, days + pred_days + seq_length)
    
    if df.empty:
        st.error(f"No data found for {crypto_symbol}. Please check the symbol and try again.")
    else:
        st.write("### Raw Data")
        st.write(df.tail())
        
        # Check for any missing values and fill them
        if df.isnull().sum().sum() > 0:
            st.warning("Some missing values detected in the data. Filling with forward fill method.")
            df = df.fillna(method='ffill')
            
        # Prepare features
        feature_data = df[features].values
        
        # Apply scaling
        if scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
            
        scaled_data = scaler.fit_transform(feature_data)
        
        # Always have Close price as the first column for prediction target
        close_idx = features.index("Close") if "Close" in features else 0
        if close_idx != 0 and len(features) > 1:
            # Reorder to have Close as first column
            scaled_data = np.hstack((scaled_data[:, close_idx:close_idx+1], 
                                     np.delete(scaled_data, close_idx, axis=1)))
        
        # Prepare data based on model choice
        if model_choice in ["LSTM", "GRU", "Bidirectional LSTM"]:
            X, y = create_sequences(scaled_data, seq_length)
        else:  # Random Forest
            X, y = create_rf_dataset(scaled_data, seq_length)
        
        # Split into train and test sets
        test_idx = int(len(X) * (1 - test_size))
        X_train_full, X_test = X[:test_idx], X[test_idx:]
        y_train_full, y_test = y[:test_idx], y[test_idx:]
        
        # Further split training data into train and validation
        if validation_split > 0:
            val_idx = int(len(X_train_full) * (1 - validation_split))
            X_train, X_val = X_train_full[:val_idx], X_train_full[val_idx:]
            y_train, y_val = y_train_full[:val_idx], y_train_full[val_idx:]
        else:
            X_train, X_val = X_train_full, None
            y_train, y_val = y_train_full, None
        
        st.write("### Training the Model...")
        
        # Different training process for neural networks vs random forest
        if model_choice in ["LSTM", "GRU", "Bidirectional LSTM"]:
            # Setup callbacks for neural networks
            callbacks = []
            if early_stopping:
                callbacks.append(EarlyStopping(monitor='val_loss' if validation_split > 0 else 'loss', 
                                             patience=patience, restore_best_weights=True))
            if reduce_lr:
                callbacks.append(ReduceLROnPlateau(monitor='val_loss' if validation_split > 0 else 'loss', 
                                                 factor=0.5, patience=patience//2, min_lr=1e-6))
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if optimize_model:
                # Hyperparameter tuning
                if tuner_type == "Random Search":
                    tuner = RandomSearch(
                        lambda hp: build_model_hp(hp, input_shape=input_shape, model_type=model_choice),
                        objective='val_loss' if validation_split > 0 else 'loss',
                        max_trials=n_trials,
                        directory='crypto_tuner',
                        project_name='crypto_optimization'
                    )
                else:  # Bayesian Optimization
                    tuner = BayesianOptimization(
                        lambda hp: build_model_hp(hp, input_shape=input_shape, model_type=model_choice),
                        objective='val_loss' if validation_split > 0 else 'loss',
                        max_trials=n_trials,
                        directory='crypto_tuner',
                        project_name='crypto_optimization'
                    )
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom callback to update progress
                class TunerProgressCallback(tf.keras.callbacks.Callback):
                    def __init__(self, total_trials):
                        self.total_trials = total_trials
                        self.current_trial = 0
                    
                    def on_train_begin(self, logs=None):
                        self.current_trial += 1
                        status_text.text(f"Running trial {self.current_trial}/{self.total_trials}")
                    
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (self.current_trial - 1 + (epoch + 1) / epochs) / self.total_trials
                        progress_bar.progress(min(progress, 1.0))
                
                validation_data = (X_val, y_val) if validation_split > 0 else None
                
                tuner.search(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    callbacks=[*callbacks, TunerProgressCallback(n_trials)]
                )
                
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = build_model_hp(best_hps, input_shape=input_shape, model_type=model_choice)
                
                st.write(f"### Best Hyperparameters Found:")
                for param, value in best_hps.values.items():
                    st.write(f"- {param}: {value}")
            else:
                model = build_simple_model(input_shape=input_shape, model_type=model_choice)
            
            # Train with the best model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val) if validation_split > 0 else None,
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history for neural networks
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
            if validation_split > 0:
                fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
            fig_history.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig_history)
            
        else:  # Random Forest model
            st.write(f"Training Random Forest with {n_estimators} trees...")
            
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=42
            )
            
            with st.spinner('Training Random Forest model...'):
                model.fit(X_train, y_train)
            
            if validation_split > 0:
                val_score = model.score(X_val, y_val)
                st.write(f"Validation R² Score: {val_score:.4f}")
            
            # Feature importance for Random Forest
            if len(features) > 1:
                feature_importances = model.feature_importances_
                
                # Calculate feature names for the flattened input
                feature_names = []
                for i in range(seq_length):
                    for feature in features:
                        feature_names.append(f"{feature} (t-{seq_length-i})")
                
                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(feature_importances)],
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False)
                
                st.write("### Feature Importance")
                st.write(importance_df.head(10))  # Show top 10 features
                
                # Plot feature importance
                fig_imp = go.Figure()
                fig_imp.add_trace(go.Bar(
                    x=importance_df['Feature'][:10],  # Top 10 features
                    y=importance_df['Importance'][:10],
                    marker_color='royalblue'
                ))
                fig_imp.update_layout(
                    title='Top 10 Feature Importance',
                    xaxis_title='Feature',
                    yaxis_title='Importance',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig_imp)
        
        # Generate predictions for test data
        test_predictions = model.predict(X_test)
        
        # For neural networks, reshape predictions
        if model_choice in ["LSTM", "GRU", "Bidirectional LSTM"]:
            test_predictions = test_predictions.flatten()
        
        # Predict on full historical data if requested
        if predict_full_history:
            historical_predictions = model.predict(X)
            
            # For neural networks, reshape predictions
            if model_choice in ["LSTM", "GRU", "Bidirectional LSTM"]:
                historical_predictions = historical_predictions.flatten()
            
            # Convert only the first column (Close price) back to original scale
            historical_pred_scaled = np.zeros_like(scaled_data)
            for i in range(len(historical_predictions)):
                historical_pred_scaled[i+seq_length, 0] = historical_predictions[i]
            
            # Only inverse transform the rows that have predictions
            historical_pred_inv = np.zeros_like(historical_pred_scaled)
            for i in range(seq_length, len(historical_pred_scaled)):
                single_pred = np.copy(scaled_data[i])
                single_pred[0] = historical_pred_scaled[i, 0]
                historical_pred_inv[i] = scaler.inverse_transform(single_pred.reshape(1, -1))
            
            # Extract just the predicted Close prices
            historical_pred_prices = historical_pred_inv[seq_length:, 0]
            
            # Create DataFrame for the historical predictions
            historical_dates = df.index[seq_length:]
            historical_pred_df = pd.DataFrame(index=historical_dates, 
                                            data={'Predicted': historical_pred_prices})
        
        # Convert test predictions to original scale
        test_pred_inv = np.zeros((len(test_predictions), feature_data.shape[1]))
        test_pred_inv[:, 0] = test_predictions  # Set predicted Close values
        test_pred_inv = scaler.inverse_transform(test_pred_inv)[:, 0]
        
        # Get actual test values
        test_actual_inv = np.zeros((len(y_test), feature_data.shape[1]))
        test_actual_inv[:, 0] = y_test  # Set actual Close values
        test_actual_inv = scaler.inverse_transform(test_actual_inv)[:, 0]
        
        # Generate future predictions
        future_predictions = []
        
        if model_choice in ["LSTM", "GRU", "Bidirectional LSTM"]:
            # Neural network future prediction
            last_sequence = X[-1:].reshape(1, seq_length, X.shape[2])  # Start with the last known sequence
            
            for _ in range(pred_days):
                # Predict the next step
                next_pred = model.predict(last_sequence)
                future_predictions.append(next_pred[0, 0])
                
                # Create a new sequence by shifting and adding the new prediction
                next_seq = np.copy(last_sequence[0, 1:, :])
                next_val = np.zeros((1, 1, X.shape[2]))
                next_val[0, 0, 0] = next_pred[0, 0]  # Set predicted Close
                
                last_sequence = np.concatenate([next_seq.reshape(1, -1, X.shape[2]), next_val], axis=1)
        else:
            # Random Forest future prediction - flatten the sequence for RF
            last_sequence = X[-1:].reshape(1, -1)  # Get the last known sequence
            
            for _ in range(pred_days):
                # Predict the next step
                next_pred = model.predict(last_sequence)
                future_predictions.append(next_pred[0])
                
                # Create a new sequence by shifting and adding the new prediction
                # First remove the oldest timestep features
                feature_count = X.shape[1] // seq_length
                new_sequence = last_sequence[0, feature_count:].copy()
                
                # Add new prediction and zeros for other features if any
                new_features = np.zeros(feature_count)
                new_features[0] = next_pred[0]  # Set predicted Close
                
                # Combine to form the new sequence
                last_sequence = np.append(new_sequence, new_features).reshape(1, -1)
        
        # Convert future predictions to original scale
        future_inverse = np.zeros((len(future_predictions), feature_data.shape[1]))
        future_inverse[:, 0] = future_predictions  # Set predicted Close values
        future_inverse = scaler.inverse_transform(future_inverse)[:, 0]
        
        # Date range for future predictions
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(future_inverse))
        future_df = pd.DataFrame(index=future_dates, data={'Predicted': future_inverse})
        
        # Evaluate test predictions
        rmse = math.sqrt(mean_squared_error(test_actual_inv, test_pred_inv))
        mae = mean_absolute_error(test_actual_inv, test_pred_inv)
        r_squared = r2_score(test_actual_inv, test_pred_inv)
        mape = mean_absolute_percentage_error(test_actual_inv, test_pred_inv)
        
        st.write("### Model Evaluation (Test Data)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("MAE", f"{mae:.4f}")
        with col2:
            st.metric("R²", f"{r_squared:.4f}")
            st.metric("MAPE", f"{mape:.2f}%")
        
        # Create results DataFrame for test data
        test_dates = df.index[test_idx+seq_length:]
        test_results_df = pd.DataFrame(
            index=test_dates,
            data={'Actual': test_actual_inv, 'Predicted': test_pred_inv}
        )
        
        # Visualization
        fig = go.Figure()
        
        # Plot historical candlestick data
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Historical Data",
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        
        if predict_full_history:
            # Plot predictions for the entire historical period
            fig.add_trace(go.Scatter(
                x=historical_pred_df.index,
                y=historical_pred_df['Predicted'],
                mode='lines',
                name='Historical Predictions',
                line=dict(color='blue', width=2)
            ))
        else:
            # Plot only test period predictions
            fig.add_trace(go.Scatter(
                x=test_results_df.index,
                y=test_results_df['Predicted'],
                mode='lines',
                name='Test Predictions',
                line=dict(color='blue', width=2)
            ))
        
        # Plot future predictions
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df['Predicted'],
            mode='lines',
            name='Future Predictions',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{crypto_symbol} Price Prediction with {model_choice}',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction results
        if predict_full_history:
            st.write("### Historical Predictions (Sample)")
            st.write(historical_pred_df.tail(10))
        
        st.write("### Test Period Predictions")
        st.write(test_results_df)
        
        st.write("### Future Predictions")
        st.write(future_df)
        
        # Additional visualization: Actual vs Predicted for test period
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=test_results_df.index,
            y=test_results_df['Actual'],
            mode='lines',
            name='Actual',
            line=dict(color='green', width=2)
        ))
        
        fig2.add_trace(go.Scatter(
            x=test_results_df.index,
            y=test_results_df['Predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='blue', width=2)
        ))
        
        fig2.update_layout(
            title=f'Test Period: Actual vs Predicted Prices ({model_choice})',
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)

        if show_recommendations:
            st.write("### Trading Recommendations")

            # Get current price
            current_price = df['Close'].iloc[-1]

            # Generate recommendations
            recommendations = generate_trading_recommendations(future_df, current_price)

            # Display overall recommendation with color coding
            rec = recommendations['overall_recommendation']
            rec_color = 'green' if 'Buy' in rec else 'red' if 'Sell' in rec else 'orange'

            st.markdown(f"<h4 style='color: {rec_color};'>Overall Recommendation: {rec}</h4>", unsafe_allow_html=True)
            st.write(f"**Summary:** {recommendations['summary']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trend Strength", f"{recommendations['trend_strength']:.2f}%")
            with col2:
                st.metric("Volatility", f"{recommendations['volatility']:.2f}%")
            with col3:
                st.metric("Risk Level", recommendations['risk_level'])

            # Display detailed recommendations in a table
            detailed_recs = recommendations['detailed_recommendations']

            # Create a DataFrame for the detailed recommendations
            rec_data = []
            for rec in detailed_recs:
                action_color = 'green' if rec['action'] == 'Buy' else 'red' if rec['action'] == 'Sell' else 'orange'
                rec_data.append({
                    'Timeframe': rec['timeframe'],
                    'Expected Return': f"{rec['expected_return']:.2f}%",
                    'Action': f"<span style='color: {action_color};'><b>{rec['action']}</b></span>",
                    'Confidence': f"{rec['confidence']:.1f}%",
                    'Reasoning': rec['reasoning']
                })

            rec_df = pd.DataFrame(rec_data)

            # Display the table
            st.write("#### Detailed Recommendations")
            st.markdown(rec_df.to_html(escape=False, index=False), unsafe_allow_html=True)

            # Adjusting recommendations based on user's risk tolerance
            st.write("#### Personalized Advice Based on Your Risk Tolerance")

            risk_mapping = {
                "Very Low": 0,
                "Low": 1,
                "Medium": 2,
                "High": 3,
                "Very High": 4
            }

            risk_score = risk_mapping[risk_tolerance]

            if risk_score <= 1:  # Very Low to Low
                if 'Sell' in rec:
                    st.write("**Considering your low risk tolerance:** You might want to consider taking profits or setting stop-loss orders to protect your capital.")
                elif 'Buy' in rec:
                    st.write("**Considering your low risk tolerance:** You might want to wait for stronger confirmation signals or invest only a small portion of your capital.")
                else:
                    st.write("**Considering your low risk tolerance:** The current neutral recommendation aligns with your risk preference. Consider stable assets or dollar-cost averaging strategies.")
            elif risk_score == 2:  # Medium
                st.write("**Considering your medium risk tolerance:** The above recommendations should be suitable for your risk profile. Consider a balanced approach to position sizing.")
            else:  # High to Very High
                if 'Buy' in rec:
                    st.write("**Considering your high risk tolerance:** You might consider more aggressive entry strategies or larger position sizes to maximize potential gains.")
                elif 'Sell' in rec:
                    st.write("**Considering your high risk tolerance:** You might consider short positions or hedging strategies to capitalize on the predicted downtrend.")
                else:
                    st.write("**Considering your high risk tolerance:** Despite the neutral recommendation, you might look for shorter-term trading opportunities based on technical indicators.")

            # Add a visualization for recommendations
            st.write("#### Visual Recommendation Timeline")

            # Create a visualization of future price predictions with buy/sell annotations
            fig_rec = go.Figure()

            # Plot future predictions
            fig_rec.add_trace(go.Scatter(
                x=future_df.index,
                y=future_df['Predicted'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='blue', width=2)
            ))

            # Add horizontal line for current price
            fig_rec.add_shape(
                type="line",
                x0=future_df.index[0],
                y0=current_price,
                x1=future_df.index[-1],
                y1=current_price,
                line=dict(color="black", width=1, dash="dash"),
            )

            # Add annotations for recommendations
            for i, rec in enumerate(detailed_recs):
                if rec['action'] != 'Hold':
                    idx = 6 if i == 0 else 13 if i == 1 else len(future_df) - 1
                    fig_rec.add_annotation(
                        x=future_df.index[idx],
                        y=future_df['Predicted'].iloc[idx],
                        text=rec['action'],
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='green' if rec['action'] == 'Buy' else 'red',
                        bgcolor='green' if rec['action'] == 'Buy' else 'red',
                        font=dict(color='white')
                    )

            fig_rec.update_layout(
                title='Future Price Prediction with Trading Recommendations',
                xaxis_title='Date',
                yaxis_title='Price',
                showlegend=True
            )

            st.plotly_chart(fig_rec, use_container_width=True)
        
        st.success(f"Prediction with {model_choice} completed!")