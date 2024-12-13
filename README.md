# Stock Price Prediction Using LSTM

This project is a stock price prediction model built using Long Short-Term Memory (LSTM) networks. The goal of the project is to predict the future stock prices of a company based on its historical stock prices. We utilize time series data and apply LSTM, a type of recurrent neural network (RNN), to model the sequential nature of the stock price data.

## Introduction


Stock market prediction is a challenging problem due to the volatile and non-linear nature of financial markets. This project focuses on using machine learning techniques to predict stock prices using historical data. Specifically, we implement a LSTM-based model, which is well-suited for time series forecasting tasks due to its ability to capture long-term dependencies in sequential data.

The stock price data is preprocessed, normalized, and used to train an LSTM model to predict the closing price of a stock for the next day. The model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics, and its predictions are compared to the actual stock prices.

### 1. **Clone the Repository**
   First, you need to clone the repository containing the project to your local machine.

   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

### 2. **Set Up the Python Environment**
   Create a virtual environment to isolate your project dependencies.

   - **Create a virtual environment**:
     ```bash
     python -m venv venv
     ```

   - **Activate the virtual environment**:
     - On **Windows**:
       ```bash
       venv\Scripts\activate
       ```
     - On **macOS/Linux**:
       ```bash
       source venv/bin/activate
       ```
### Step 3: Install Dependencies

Before running the project, you'll need to install the necessary dependencies. The required packages are listed in the `requirements.txt` file. To install them, run the following command in your terminal:

```bash
pip install -r requirements.txt

