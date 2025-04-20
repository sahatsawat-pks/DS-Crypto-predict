# 🚀 Cryptocurrency Trend Prediction

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A powerful and user-friendly web application for predicting cryptocurrency price trends using advanced machine learning techniques. This tool leverages LSTM neural networks, GRU, and Random Forest models to analyze historical data and generate forecasts with interactive visualizations.

## 📊 Demo
![Demo Screenshot](<demo/Demo app.png>)

## ✨ Key Features

- **Multiple ML Models**: Choose between LSTM, Bidirectional LSTM, GRU, and Random Forest algorithms
- **Real-time Data Acquisition**: Fetches up-to-date cryptocurrency data directly from **yFinance**
- **Flexible Configuration**:
  - Training window: 60-730 days of historical data
  - Prediction horizon: 7-365 days into the future
  - Sequence length: 5-60 days (lookback window)
- **Advanced Options**:
  - Bidirectional LSTM layers for complex pattern detection
  - Multiple scalers (MinMax, Standard, Robust)
  - Hyperparameter optimization with Keras Tuner
- **Interactive Visualizations**: Beautiful Plotly candlestick charts showing:
  - Historical price data
  - Model predictions on historical data
  - Future price projections with realistic volatility
- **Performance Metrics**: RMSE, MAE, MAPE, and R-squared to evaluate model accuracy
- **Trading Recommendations**: Actionable insights based on predictions and risk tolerance
- **User-Friendly Interface**: Clean, intuitive design powered by Streamlit

## 🛠️ Installation

### Prerequisites

- Python 3.6+
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cryptocurrency-trend-prediction.git
   cd cryptocurrency-trend-prediction
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface** at `http://localhost:8501` in your browser

### Using the Application

1. **Basic Setup**:
   - Enter a cryptocurrency symbol (e.g., `BTC-USD`, `ETH-USD`, `SOL-USD`)
   - Select training and prediction periods
   - Choose sequence length for time series analysis

2. **Advanced Configuration**:
   - Select your preferred model type
   - Toggle bidirectional LSTM (if applicable)
   - Choose scaling method
   - Set batch size and epochs
   - Enable hyperparameter optimization (optional)

3. **Get Results**:
   - Click "Train and Predict" to start the process
   - Review model performance metrics
   - Analyze the interactive charts
   - Consider the trading recommendations

## 📊 Model Details

### Available Models

- **LSTM (Long Short-Term Memory)**: Excellent for capturing long-term dependencies in time series data
- **Bidirectional LSTM**: Processes sequences in both forward and backward directions
- **GRU (Gated Recurrent Unit)**: Similar to LSTM but with a simpler architecture
- **Random Forest**: Ensemble learning method for regression tasks

### Optimization

The application optionally uses **Keras Tuner** to find optimal hyperparameters:
- Number of LSTM/GRU units
- Dropout rates
- Learning rates
- Number of layers
- Optimizer selection

## 📋 Project Structure

```
📂 Cryptocurrency-Trend-Prediction
│── app.py            # Main Streamlit application
│── requirements.txt  # Python dependencies
│── README.md         # Project documentation
│── crypto_tuner/     # Directory for tuning results
```

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Cryptocurrency markets are highly volatile, and no prediction tool can guarantee future results. Never make investment decisions based solely on these predictions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.