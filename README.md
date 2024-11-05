# Brent Oil Prices Analysis and Dashboard Development

## Task 1: Time Series Analysis and Basic EDA

### Overview

In Task 1, we perform initial exploration of the historical Brent oil prices dataset to understand trends, patterns, and any anomalies present in the data.

### Data Loading and Preprocessing

```python
import pandas as pd

# Load the dataset
data_path = "../data/BrentOilPrices.csv"
df = pd.read_csv(data_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Display the first few rows
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Plot Brent oil prices over time
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x=df.index, y='Price')
plt.title('Brent Oil Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD per Barrel)')
plt.show()


### Key Insights

- The line plot reveals significant fluctuations in oil prices over the decades.
- Volatility corresponds with major geopolitical and economic events.


## Task 2: Advanced Time Series and Econometric Modeling

### Overview

In Task 2, we refine the analysis by applying advanced time series models, such as ARIMA, GARCH, Markov-Switching ARIMA, and LSTM, to better understand trends, volatility, and regime changes in Brent oil prices.

### ARIMA Model Implementation

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
arima_model = ARIMA(df['Price'], order=(5, 1, 0))
arima_fit = arima_model.fit()

# Summary of the ARIMA model
print(arima_fit.summary())

from arch import arch_model

# Fit GARCH model
price_diff = df['Price'].diff().dropna()
garch_model = arch_model(price_diff, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Summary of the GARCH model
print(garch_fit.summary())


from arch import arch_model

# Fit GARCH model
price_diff = df['Price'].diff().dropna()
garch_model = arch_model(price_diff, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Summary of the GARCH model
print(garch_fit.summary())


from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Fit Markov-Switching ARIMA model
ms_arima = MarkovRegression(df['Price'], k_regimes=2, trend='c', switching_variance=True)
ms_arima_fit = ms_arima.fit()

# Summary of the Markov-Switching ARIMA model
print(ms_arima_fit.summary())


### LSTM Model Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Prepare data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

def prepare_lstm_data(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

X, y = prepare_lstm_data(scaled_data, 10)

# Build LSTM model
model_lstm = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X, y, epochs=20, batch_size=32)


### Key Insights

- **ARIMA**: Captures linear trends effectively but struggles with volatility.
- **GARCH**: Excellent for modeling and forecasting volatility.
- **Markov-Switching ARIMA**: Highlights regime shifts, correlating with major market events.
- **LSTM**: Handles complex, non-linear patterns and dependencies.


## Task 3: Developing an Interactive Dashboard

### Overview

In Task 3, we build an interactive dashboard using Flask for the backend and React for the frontend. This dashboard visualizes the results of the Brent oil prices analysis, allowing users to interact with the data through dynamic visualizations and filters.

### Flask Backend

#### Setting Up Flask and API Endpoints

```python
from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

# Load the dataset
data_path = "../data/BrentOilPrices.csv"
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

# API to get historical data
@app.route('/api/data', methods=['GET'])
def get_data():
    start_date = request.args.get('start', default='1987-05-20')
    end_date = request.args.get('end', default='2022-09-30')
    filtered_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return jsonify(filtered_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)


import React, { useState, useEffect } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import { TextField, Button, Container, Typography } from '@mui/material';

function App() {
    const [data, setData] = useState([]);
    const [startDate, setStartDate] = useState('1987-05-20');
    const [endDate, setEndDate] = useState('2022-09-30');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = () => {
        setLoading(true);
        fetch(`http://localhost:5000/api/data?start=${startDate}&end=${endDate}`)
            .then(response => response.json())
            .then(data => {
                setData(data);
                setLoading(false);
            })
            .catch(error => {
                setError(error);
                setLoading(false);
            });
    };

    const handleDateSubmit = (e) => {
        e.preventDefault();
        fetchData();
    };

    return (
        <Container className="App">
            <Typography variant="h4" gutterBottom>
                Brent Oil Prices Dashboard
            </Typography>
            <form onSubmit={handleDateSubmit} className="form">
                <TextField
                    label="Start Date"
                    type="date"
                    value={startDate}
                    onChange={e => setStartDate(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    variant="outlined"
                    style={{ marginRight: '20px' }}
                />
                <TextField
                    label="End Date"
                    type="date"
                    value={endDate}
                    onChange={e => setEndDate(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    variant="outlined"
                />
                <Button type="submit" variant="contained" color="primary" style={{ marginLeft: '20px' }}>
                    Fetch Data
                </Button>
            </form>
            {loading ? <Typography>Loading...</Typography> : (
                <LineChart width={800} height={400} data={data}>
                    <Line type="monotone" dataKey="Price" stroke="#1976d2" />
                    <CartesianGrid stroke="#ccc" />
                    <XAxis dataKey="Date" />
                    <YAxis />
                    <Tooltip />
                </LineChart>
            )}
            {error && <Typography color="error">Error loading data: {error.message}</Typography>}
        </Container>
    );
}

export default App;



Key Features
Dynamic Date Filtering: Users can input custom start and end dates to fetch and visualize specific data ranges.
Interactive Visualization: Line chart displays historical trends, with tooltips for detailed information.
Enhanced UI: Utilizes Material-UI for a modern, responsive design.