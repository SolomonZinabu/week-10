from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
 
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Allow CORS for localhost:3000  might need to change to allow all for testing purposes

# Load data and models
data_path = "data/BrentOilPrices.csv"
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

ms_arima_model = joblib.load("models/ms_arima_model.pkl")
lstm_model = load_model("models/lstm_model.h5")

# API to get historical data
@app.route('/api/data', methods=['GET'])
def get_data():
    start_date = request.args.get('start', default='1987-05-20')
    end_date = request.args.get('end', default='2022-09-30')
    data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return jsonify(data.to_dict(orient='records'))

# API to get model predictions
@app.route('/api/predict', methods=['GET'])
def get_predictions():
    # Dummy response for model prediction
    return jsonify({"message": "Model predictions endpoint"})

if __name__ == '__main__':
    app.run(debug=True)
