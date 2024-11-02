# data_analysis_workflow.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

def load_data(file_path):
    """Load Brent Oil Prices data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data by parsing dates and handling missing values."""
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df.dropna(inplace=True)
    return df

def plot_price_trend(df):
    """Plot the trend of Brent oil prices over time."""
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='Date', y='Price')
    plt.title('Brent Oil Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.show()

def fit_arima_model(df):
    """Fit an ARIMA model to the Brent oil price data."""
    df.set_index('Date', inplace=True)  # ARIMA requires a datetime index
    model = ARIMA(df['Price'], order=(5,1,0))  # Example order; should be tuned
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def plot_residuals(model_fit):
    """Plot residuals to understand the model's performance."""
    residuals = model_fit.resid
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals of ARIMA Model')
    plt.xlabel("Residuals")
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    data_path = "../data/BrentOilPrices.csv"
    df = load_data(data_path)
    df = preprocess_data(df)
    
    # Plot price trend
    plot_price_trend(df)
    
    # Fit ARIMA model and plot residuals
    model_fit = fit_arima_model(df)
    plot_residuals(model_fit)
