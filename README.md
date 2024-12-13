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

### Step 4: Load the Dataset

The project requires a dataset of historical stock prices for training and testing the model. Make sure the dataset is in the correct format, typically a CSV file containing stock prices, including the 'Close' price for each day.

1. Download the dataset (or use your own).
2. Ensure that the dataset has the necessary columns, such as `Date` and `Close`.
3. Load the dataset into the project by running the following code:

```python
import pandas as pd

# Load your dataset
data = pd.read_csv('path/to/your/dataset.csv')

# Check the first few rows of the dataset
data.head()


```bash

pip install -r requirements.txt

```
### Step 5: Preprocess the Data

Before training the model, you need to preprocess the data by normalizing it and preparing it for model input.

1. **Normalize the Data:**
   Normalize the stock price data to a range between 0 and 1 using `MinMaxScaler`. This step helps the model to learn more effectively.

2. **Prepare the Data for Training:**
   Split the data into training and testing datasets. You will also need to create sequences of past stock prices to predict future prices.

Run the following code to preprocess the data:

```python
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalize the 'Close' price data
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create the training and testing datasets
train_size = int(len(data) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Prepare sequences for training (use the past 100 days to predict the next day)
x_train = []
y_train = []
for i in range(100, len(train_data)):
    x_train.append(train_data[i-100:i])
    y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Similarly, prepare the test data
x_test = []
y_test = []
for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i])
    y_test.append(test_data[i, 0])
```
### Step 6: Build the Model

Now that the data is preprocessed, you can proceed to build the model. In this step, we will create an LSTM (Long Short-Term Memory) model, which is ideal for time series forecasting like stock prices.

1. **Define the Model Architecture:**
   We will use an LSTM layer followed by Dense layers. LSTM is suitable for sequence data and will help in predicting future stock prices based on past data.

Run the following code to define the model architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Initialize the model
model = Sequential()

# Add an LSTM layer with 50 units and return sequences to the next LSTM layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

# Add a Dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Add another LSTM layer
model.add(LSTM(units=50, return_sequences=False))

# Add a Dropout layer
model.add(Dropout(0.2))

# Add a Dense layer for the output
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
model.summary()

x_test = np.array(x_test)
y_test = np.array(y_test)
```
