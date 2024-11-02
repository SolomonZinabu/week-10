# Brent Oil Price Analysis - Interim Submission

This project analyzes historical Brent oil prices to detect significant changes and understand how political and economic events affect price trends. The analysis is part of Birhan Energies’ mission to support stakeholders in the energy sector with data-driven insights.

## Project Structure

```plaintext
project/
│
├── data/
│   └── BrentOilPrices.csv       # Dataset containing historical Brent oil prices
│
├── notebooks/
│   └── task-1.ipynb             # Notebook covering Task 1 (Data Analysis Workflow & EDA)
│
└── scripts/
    └── data_analysis_workflow.py # Python script with data loading, preprocessing, and ARIMA model functions
Data Summary
The dataset (BrentOilPrices.csv) contains daily Brent oil prices from May 20, 1987, to September 30, 2022, with two main columns:

Date: Date of the price record
Price: Price of Brent oil in USD per barrel
Requirements
To run the code, install the necessary packages:

pip install pandas matplotlib seaborn statsmodels
Analysis Workflow
Load Data: Load and preprocess data to ensure consistent date formats and handle any missing values.
Exploratory Data Analysis (EDA): Visualize the time series data to identify trends and fluctuations.
ARIMA Model: Fit an ARIMA model to analyze price trends and detect changes.
Change Point Detection: Identify periods of significant changes in the oil prices (to be further explored).
Key Files
1. notebooks/task-1.ipynb
The main notebook for defining the data analysis workflow, performing EDA, and fitting an ARIMA model.

Code Snippet: Loading Data and Initial EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = "../data/BrentOilPrices.csv"
df = pd.read_csv(data_path)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

# Plot the price trend
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Date', y='Price')
plt.title('Brent Oil Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD per Barrel)')
plt.show()


2. scripts/data_analysis_workflow.py
This script includes functions for data preprocessing, visualization, and ARIMA model fitting.

Code Snippet: ARIMA Model Fitting and Residuals Analysis

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

def fit_arima_model(df):
    """Fit an ARIMA model to the Brent oil price data."""
    df.set_index('Date', inplace=True)
    model = ARIMA(df['Price'], order=(5,1,0))  # Order can be tuned
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def plot_residuals(model_fit):
    """Plot residuals to assess model performance."""
    residuals = model_fit.resid
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals of ARIMA Model')
    plt.xlabel("Residuals")
    plt.show()

# Load data and run ARIMA model
data_path = "../data/BrentOilPrices.csv"
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

# Fit ARIMA model
model_fit = fit_arima_model(df)
plot_residuals(model_fit)


Model Summary
Our ARIMA(5,1,0) model provided initial insights into the price trend. Here’s a brief summary:

AIC: 29093.226
BIC: 29135.862
Standard Error (Sigma): 1.4767
Key Findings
The model residuals indicate some potential volatility in the data.
High kurtosis and skewness suggest that certain periods experience heightened volatility, likely linked to external events.
Next Steps
For the final submission, we plan to:

Improve Model Accuracy: Explore alternative models like GARCH to better handle volatility clustering.
Change Point Analysis: Implement Bayesian change point detection for identifying major shifts in price trends.
Dashboard Development: Visualize results through an interactive dashboard for stakeholder engagement.