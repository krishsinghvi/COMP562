# Stock Price Prediction Using LSTM

This project is a stock price prediction model built using Long Short-Term Memory (LSTM) networks. The goal of the project is to predict the future stock prices of a company based on its historical stock prices. We utilize time series data and apply LSTM, a type of recurrent neural network (RNN), to model the sequential nature of the stock price data.

## Introduction


Stock market prediction is a challenging problem due to the volatile and non-linear nature of financial markets. This project focuses on using machine learning techniques to predict stock prices using historical data. Specifically, we implement a LSTM-based model, which is well-suited for time series forecasting tasks due to its ability to capture long-term dependencies in sequential data.

The stock price data is preprocessed, normalized, and used to train an LSTM model to predict the closing price of a stock for the next day. The model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics, and its predictions are compared to the actual stock prices.

Technologies Used
Python: The primary programming language for implementing the machine learning model and preprocessing the data.
Keras: A high-level neural networks API, used for building and training the LSTM model.
TensorFlow: Backend framework for Keras, used for training the model.
Pandas: Data manipulation library used for loading and preprocessing the stock price data.
NumPy: Used for numerical computations and handling arrays.
Matplotlib: A library for creating static, animated, and interactive visualizations in Python.
Scikit-learn: A library for machine learning, used for scaling and evaluation metrics (e.g., MSE, RMSE).
Jupyter Notebook: Used for running the code and visualizing the results interactively.
Dataset
The dataset used in this project consists of historical stock prices of a company. The dataset includes the following columns:

# Stock Price Prediction Using LSTM

This project demonstrates how to predict stock prices using a Long Short-Term Memory (LSTM) model. The LSTM model is trained on historical stock price data and used to forecast future prices. The project involves data preprocessing, model training, and evaluation using various metrics.

## Steps to Get the Project Working

### 1. **Clone the Repository**
   First, you need to clone the repository containing the project to your local machine.

   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
