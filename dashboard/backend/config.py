import os
import sys
import pickle
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import yfinance as yf

# Add the 'data' directory as one where we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Define paths
DATA_PATH = '../../data/BrentOilPrices.csv'
MODEL_PATH = './models/lstm_model.h5'       
SCALER_PATH = './models/scaler.pkl'
MERGE_PATH = '../../data/merged_data.csv'

# Load Brent oil price data
df = pd.read_csv(DATA_PATH)

# Load the LSTM model and scaler
model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as file:
    scaler = pickle.load(file)

# Function to fetch recent Brent oil prices from yfinance
def fetch_recent_brent_prices(ticker="BZ=F", period="1y", interval="1d"):
    """
    Fetch recent Brent oil prices from Yahoo Finance.
    - ticker: Yahoo Finance ticker for Brent Oil Futures, typically "BZ=F"
    - period: Time period (e.g., '1y' for 1 year)
    - interval: Data interval (e.g., '1d' for daily prices)
    """
    brent_data = yf.download(ticker, period=period, interval=interval)
    brent_data.reset_index(inplace=True)
    brent_data.rename(columns={'Close': 'Price'}, inplace=True)
    return brent_data[['Date', 'Price']]