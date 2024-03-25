import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load the sales data
sales_data = pd.read_csv('sales_data.csv')

# Convert the date column to datetime type
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

# Set the Date column as the index
# Set the Date column as the index with explicit frequency
sales_data.set_index('Date', inplace=True)
sales_data.index.freq = 'D'

# Explore the data
print(sales_data.head())

# Visualize the sales data
sales_data['Sales'].plot(figsize=(12, 6))
plt.title('Sales Data Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Decompose the time series into trend, seasonality, and residuals
decomposition = seasonal_decompose(sales_data['Sales'], model='additive')

# Plot the decomposed components
decomposition.plot()
plt.suptitle('Decomposition of Sales Data')
plt.show()

# Split data into train and test sets
train_size = int(len(sales_data) * 0.8)
train_data, test_data = sales_data.iloc[:train_size], sales_data.iloc[train_size:]

# Build and train the forecasting model (Holt-Winters Exponential Smoothing)
model = ExponentialSmoothing(train_data['Sales'], seasonal='add', seasonal_periods=12).fit()

# Forecast future demand
forecast = model.forecast(len(test_data))

# Evaluate the model
mse = mean_squared_error(test_data['Sales'], forecast)
print(f'Mean Squared Error: {mse}')

# Visualize the forecast
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Sales'], label='Train')
plt.plot(test_data.index, test_data['Sales'], label='Test')
plt.plot(test_data.index, forecast, label='Forecast')
plt.title('Demand Forecasting using Holt-Winters Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
