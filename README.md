# Demand Forecasting for a Retail Store

## Description
This project aims to develop a time series forecasting model to predict the demand for products in a retail store using historical sales data. The model utilizes Holt-Winters Exponential Smoothing to capture seasonality and trends in the sales data.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn

## How to Use
1. Ensure you have Python installed on your system.
2. Install the required libraries by running `pip install Pandas' ; 'pip install NumPy' ; 'pip install Matplotlib' ; pip install Statsmodels; pip install Scikit-learn`.
3. Download the dataset file or replace it with your own dataset.
4. Run the code in a Python environment or Jupyter Notebook.

## Code Explanation
- Load the sales data from dataset.
- Convert the date column to datetime type and set it as the index with explicit frequency.
- Visualize the sales data over time.
- Decompose the time series into trend, seasonality, and residuals using seasonal decomposition.
- Split the data into train and test sets.
- Build and train the forecasting model using Holt-Winters Exponential Smoothing.
- Forecast future demand and evaluate the model's performance using mean squared error (MSE).
- Visualize the forecasted demand along with the actual sales data.

- ## Results
The mean squared error (MSE) for the forecasting model is calculated.
