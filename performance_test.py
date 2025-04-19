import pandas as pd
import numpy as np
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
import math
import time
import warnings
warnings.filterwarnings('ignore')

def load_data(symbol, period, interval="1d"):
    """Load cryptocurrency data from Yahoo Finance"""
    print(f"Loading data for {symbol} over {period} days...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{period}d", interval=interval)
    return df

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Target is always the Close price
    return np.array(X), np.array(y)

def create_rf_dataset(data, seq_length):
    """Create a flattened dataset for Random Forest"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].flatten())
        y.append(data[i+seq_length, 0])  # Target is always the Close price
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
    """Build a simple LSTM model"""
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_gru_model(input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
    """Build a simple GRU model"""
    model = Sequential()
    model.add(GRU(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def build_bidirectional_lstm_model(input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
    """Build a simple Bidirectional LSTM model"""
    model = Sequential()
    model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units, return_sequences=False)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    """Calculate metrics for model evaluation"""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def run_experiment(symbol, days, model_configs, seq_lengths, feature_sets, scalers, test_sizes, val_splits):
    """Run a full experiment with multiple configurations"""
    
    # Load data
    df = load_data(symbol, days + 60)  # Add extra days for longer sequences
    
    if df.empty:
        print(f"No data found for {symbol}")
        return None
    
    # Fill missing values if any
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill')
    
    results = []
    
    # Loop through all combinations
    for seq_length in seq_lengths:
        for features in feature_sets:
            feature_data = df[features].values
            
            for scaler_name, scaler_class in scalers.items():
                scaler = scaler_class()
                scaled_data = scaler.fit_transform(feature_data)
                
                # Always have Close price as the first column
                close_idx = features.index("Close") if "Close" in features else 0
                if close_idx != 0 and len(features) > 1:
                    scaled_data = np.hstack((scaled_data[:, close_idx:close_idx+1], 
                                            np.delete(scaled_data, close_idx, axis=1)))
                
                for test_size in test_sizes:
                    for val_split in val_splits:
                        for model_config in model_configs:
                            model_name = model_config['name']
                            
                            print(f"\nTesting {model_name} with seq_length={seq_length}, features={features}, " 
                                  f"scaler={scaler_name}, test_size={test_size}, val_split={val_split}")
                            
                            # Prepare data based on model type
                            if model_name != "Random Forest":
                                X, y = create_sequences(scaled_data, seq_length)
                            else:
                                X, y = create_rf_dataset(scaled_data, seq_length)
                            
                            # Split data
                            test_idx = int(len(X) * (1 - test_size))
                            X_train_full, X_test = X[:test_idx], X[test_idx:]
                            y_train_full, y_test = y[:test_idx], y[test_idx:]
                            
                            if val_split > 0:
                                val_idx = int(len(X_train_full) * (1 - val_split))
                                X_train, X_val = X_train_full[:val_idx], X_train_full[val_idx:]
                                y_train, y_val = y_train_full[:val_idx], y_train_full[val_idx:]
                            else:
                                X_train, X_val = X_train_full, None
                                y_train, y_val = y_train_full, None
                            
                            start_time = time.time()
                            
                            # Train the model
                            if model_name == "Random Forest":
                                model = RandomForestRegressor(
                                    n_estimators=model_config.get('n_estimators', 100),
                                    max_depth=model_config.get('max_depth', None),
                                    min_samples_split=model_config.get('min_samples_split', 2),
                                    min_samples_leaf=model_config.get('min_samples_leaf', 1),
                                    n_jobs=-1,
                                    random_state=42
                                )
                                model.fit(X_train, y_train)
                                
                            else:  # Neural network models
                                # Set up callbacks
                                callbacks = []
                                early_stopping = model_config.get('early_stopping', True)
                                reduce_lr = model_config.get('reduce_lr', True)
                                
                                if early_stopping:
                                    callbacks.append(EarlyStopping(
                                        monitor='val_loss' if val_split > 0 else 'loss',
                                        patience=model_config.get('patience', 10),
                                        restore_best_weights=True
                                    ))
                                    
                                if reduce_lr:
                                    callbacks.append(ReduceLROnPlateau(
                                        monitor='val_loss' if val_split > 0 else 'loss',
                                        factor=0.5,
                                        patience=model_config.get('patience', 10) // 2,
                                        min_lr=1e-6
                                    ))
                                
                                # Build the model
                                input_shape = (X_train.shape[1], X_train.shape[2])
                                units = model_config.get('units', 50)
                                dropout_rate = model_config.get('dropout_rate', 0.2)
                                optimizer = model_config.get('optimizer', 'adam')
                                    
                                if model_name == "LSTM":
                                    model = build_lstm_model(input_shape, units, dropout_rate, optimizer)
                                elif model_name == "GRU":
                                    model = build_gru_model(input_shape, units, dropout_rate, optimizer)
                                elif model_name == "Bidirectional LSTM":
                                    model = build_bidirectional_lstm_model(input_shape, units, dropout_rate, optimizer)
                                
                                # Train the model
                                history = model.fit(
                                    X_train, y_train,
                                    epochs=model_config.get('epochs', 50),
                                    batch_size=model_config.get('batch_size', 32),
                                    validation_data=(X_val, y_val) if val_split > 0 else None,
                                    callbacks=callbacks,
                                    verbose=0
                                )
                            
                            training_time = time.time() - start_time
                            
                            # Make predictions
                            test_predictions = model.predict(X_test)
                            
                            # For neural networks, reshape predictions
                            if model_name != "Random Forest":
                                test_predictions = test_predictions.flatten()
                            
                            # Convert to original scale for evaluation
                            test_pred_inv = np.zeros((len(test_predictions), feature_data.shape[1]))
                            test_pred_inv[:, 0] = test_predictions
                            test_pred_inv = scaler.inverse_transform(test_pred_inv)[:, 0]
                            
                            test_actual_inv = np.zeros((len(y_test), feature_data.shape[1]))
                            test_actual_inv[:, 0] = y_test
                            test_actual_inv = scaler.inverse_transform(test_actual_inv)[:, 0]
                            
                            # Evaluate the model
                            metrics = evaluate_model(test_actual_inv, test_pred_inv)
                            
                            # Store results
                            result = {
                                'model': model_name,
                                'seq_length': seq_length,
                                'features': '_'.join(features),
                                'feature_count': len(features),
                                'scaler': scaler_name,
                                'test_size': test_size,
                                'val_split': val_split,
                                'training_time': training_time,
                                **metrics
                            }
                            
                            # Add additional model-specific parameters
                            if model_name != "Random Forest":
                                result.update({
                                    'units': units,
                                    'dropout_rate': dropout_rate,
                                    'optimizer': optimizer,
                                    'batch_size': model_config.get('batch_size', 32),
                                    'epochs': model_config.get('epochs', 50),
                                })
                            else:
                                result.update({
                                    'n_estimators': model_config.get('n_estimators', 100),
                                    'max_depth': model_config.get('max_depth', None),
                                })
                            
                            results.append(result)
                            
                            print(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by RMSE (lower is better)
    results_df = results_df.sort_values(by='rmse')
    
    return results_df

def main():
    # Define cryptocurrency symbol and period
    symbol = "BTC-USD"  # Change to your desired cryptocurrency
    days = 730  # 2 years of data
    
    # Define model configurations
    model_configs = [
        {
            'name': 'LSTM',
            'units': 64,
            'dropout_rate': 0.2,
            'optimizer': 'adam',
            'batch_size': 32,
            'epochs': 50,
            'early_stopping': True,
            'reduce_lr': True,
            'patience': 10
        },
        {
            'name': 'GRU',
            'units': 64,
            'dropout_rate': 0.2,
            'optimizer': 'adam',
            'batch_size': 32,
            'epochs': 50,
            'early_stopping': True,
            'reduce_lr': True,
            'patience': 10
        },
        {
            'name': 'Bidirectional LSTM',
            'units': 64,
            'dropout_rate': 0.3,
            'optimizer': 'adam',
            'batch_size': 32,
            'epochs': 50,
            'early_stopping': True,
            'reduce_lr': True,
            'patience': 10
        },
        {
            'name': 'Random Forest',
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
    ]
    
    # Define sequence lengths to test
    seq_lengths = [10, 20, 30]
    
    # Define feature sets to test
    feature_sets = [
        ['Close'],
        ['Close', 'Volume'],
        ['Close', 'Open', 'High', 'Low', 'Volume']
    ]
    
    # Define scalers to test
    scalers = {
        'MinMaxScaler': MinMaxScaler,
        'StandardScaler': StandardScaler,
        'RobustScaler': RobustScaler
    }
    
    # Define test sizes to test
    test_sizes = [0.2]
    
    # Define validation splits to test
    val_splits = [0.2]
    
    # Run the experiment
    results_df = run_experiment(
        symbol, 
        days, 
        model_configs, 
        seq_lengths, 
        feature_sets, 
        scalers, 
        test_sizes, 
        val_splits
    )
    
    # Print the best configurations
    print("\n--- TOP 5 MODEL CONFIGURATIONS ---")
    print(results_df.head(5).to_string())
    
    # Save results to CSV
    results_df.to_csv(f"{symbol.replace('-', '_')}_model_comparison.csv", index=False)
    print(f"\nResults saved to {symbol.replace('-', '_')}_model_comparison.csv")
    
    # Create a detailed report of the best model
    best_model = results_df.iloc[0]
    print("\n--- BEST MODEL DETAILS ---")
    print(f"Model Type: {best_model['model']}")
    print(f"Sequence Length: {best_model['seq_length']}")
    print(f"Features: {best_model['features']}")
    print(f"Scaler: {best_model['scaler']}")
    print(f"Test Size: {best_model['test_size']}")
    print(f"Validation Split: {best_model['val_split']}")
    
    # Print model-specific parameters
    if best_model['model'] != "Random Forest":
        print(f"Units: {best_model['units']}")
        print(f"Dropout Rate: {best_model['dropout_rate']}")
        print(f"Optimizer: {best_model['optimizer']}")
        print(f"Batch Size: {best_model['batch_size']}")
        print(f"Epochs: {best_model['epochs']}")
    else:
        print(f"Number of Estimators: {best_model['n_estimators']}")
        print(f"Max Depth: {best_model['max_depth']}")
    
    # Print performance metrics
    print("\n--- PERFORMANCE METRICS ---")
    print(f"RMSE: {best_model['rmse']:.4f}")
    print(f"MAE: {best_model['mae']:.4f}")
    print(f"R²: {best_model['r2']:.4f}")
    print(f"MAPE: {best_model['mape']:.4f}")
    print(f"Training Time: {best_model['training_time']:.2f} seconds")

if __name__ == "__main__":
    main()