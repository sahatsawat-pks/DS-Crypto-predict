# Cryptocurrency Trend Prediction

This project is a **Cryptocurrency Trend Prediction** tool built with **Streamlit, TensorFlow, yFinance, and Keras Tuner**. It empowers users to train sophisticated LSTM models on historical cryptocurrency price data and generate future price predictions.

## ✨ Key Features

- **Real-time Data Acquisition:** Fetches up-to-date cryptocurrency data directly from **yFinance**.
- **Flexible Data Window:** Allows users to select the number of past days for training, ranging from 60 to 730 days.
- **Customizable Prediction Horizon:** Enables users to define the number of future days to predict, from 7 to 365 days.
- **Sequence Length Control:** Provides a slider to adjust the sequence length (look-back window) for the LSTM model, ranging from 5 to 60 days.
- **Bidirectional LSTM Option:** Offers the choice to utilize Bidirectional LSTM layers for potentially capturing more complex patterns.
- **Scalability Options:** Supports both **MinMaxScaler** and **StandardScaler** for data normalization.
- **Hyperparameter Optimization (Optional):** Integrates **Keras Tuner** for automated hyperparameter tuning of the LSTM model, potentially leading to improved performance. Users can specify the number of tuning trials.
- **Adjustable Training Parameters:** Allows users to control the number of training epochs and the batch size.
- **Interactive Visualizations:** Presents historical and predicted prices using interactive **Plotly** candlestick charts with clear distinctions between actual historical data, historical predictions, and future projections.
- **Comprehensive Evaluation Metrics:** Displays key performance metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared to assess the model's accuracy.
- **User-Friendly Interface:** Built with **Streamlit** for an intuitive and interactive web application experience.

## ⚙️ Installation

### Prerequisites

Ensure you have Python 3.6 or higher installed. You'll also need the following Python libraries. It's recommended to create a virtual environment to manage dependencies.

1.  **Install Dependencies:** Open your terminal or command prompt and run the following command to install all necessary libraries:

    ```bash
    pip install streamlit numpy pandas yfinance tensorflow scikit-learn matplotlib plotly keras-tuner
    ```

## 🚀 Usage

1.  **Save the Code:** Save the provided Python code (starting with `import streamlit as st`) as a file named `app.py`.
2.  **Navigate to the Directory:** Open your terminal or command prompt and navigate to the directory where you saved `app.py`.
3.  **Run the Streamlit App:** Execute the following command:

    ```bash
    streamlit run app.py
    ```

    This will automatically open the application in your web browser.

### How to Use the Application

1.  **Enter Cryptocurrency Symbol:** In the sidebar, type the ticker symbol of the cryptocurrency you want to analyze (e.g., `BTC-USD` for Bitcoin against USD).
2.  **Select Training Days:** Use the slider to choose the number of past days the model will use for training. A longer period might capture broader trends but could also include irrelevant data.
3.  **Select Prediction Days:** Use the slider to specify how many future days you want the model to predict.
4.  **Select Sequence Length:** Adjust the sequence length slider. This determines how many preceding days the model considers when making a prediction for the next day.
5.  **Choose Bidirectional LSTM (Optional):** Check the box if you want to use Bidirectional LSTM layers. This can sometimes improve performance by processing sequences in both forward and backward directions.
6.  **Select Scaling Method:** Choose between `MinMaxScaler` (scales data to the range [0, 1]) and `StandardScaler` (standardizes data to have zero mean and unit variance).
7.  **Optimize Model (Optional):** If you check this box, the application will use **Keras Tuner** to automatically search for the best hyperparameters for the LSTM model. This process can take some time.
8.  **Number of Tuning Trials (if optimizing):** If you've chosen to optimize the model, use this slider to set the number of different hyperparameter combinations **Keras Tuner** will try. More trials can potentially lead to better results but will take longer.
9.  **Number of Training Epochs:** Adjust the number of epochs the model will train for. More epochs can lead to better learning but also risk overfitting.
10. **Batch Size:** Select the batch size, which determines the number of samples processed before the model's weights are updated.
11. **Click "Train and Predict":** Once you've configured the parameters, click this button to start the data loading, model training, prediction, and visualization process.

### Interpreting the Results

-   **Raw Data:** Displays the last few rows of the fetched historical data.
-   **Price Prediction:** Shows a table comparing the actual closing prices with the model's predictions for the last `pred_days`.
-   **Candlestick Chart with Predictions:** An interactive candlestick chart visualizing the historical price action along with the model's historical predictions (orange line) and future price predictions (dashed red line).
-   **Prediction Evaluation:** Presents the RMSE, MAE, and R-squared scores, which provide insights into the model's prediction accuracy. Lower RMSE and MAE, and an R-squared closer to 1, generally indicate better performance.

## 🛠️ Project Structure

📂 Cryptocurrency-Trend-Prediction
│── app.py            # Main Streamlit application code
│── requirements.txt  # List of Python dependencies (you can create this with pip freeze > requirements.txt)
│── README.md         # Project documentation (this file)
│── crypto_tuner/     # Directory where Keras Tuner saves tuning results (if optimization is enabled)


## 🧠 Model Details

-   The core of the prediction engine is a **Long Short-Term Memory (LSTM)** neural network, well-suited for time series data.
-   The model architecture can be either a simple LSTM network or an optimized one using **Keras Tuner**.
-   The optimizer used is typically **Adam**, and the loss function is **Mean Squared Error (MSE)**, common choices for regression tasks like price prediction.
-   Hyperparameters such as the number of LSTM units, dropout rates, learning rate, and the number of layers can be automatically tuned if the "Optimize Model" option is selected.

**Disclaimer:** Cryptocurrency price prediction is an inherently complex and uncertain task. The results generated by this tool should not be considered financial advice. Market conditions, news events, and various other factors can significantly impact cryptocurrency prices. Use this tool for educational and analytical purposes only.