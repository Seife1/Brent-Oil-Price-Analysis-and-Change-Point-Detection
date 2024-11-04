from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from statsmodels.tsa.api import VAR
import pandas as pd

def fit_var(df):
    """Fit a Vector Autoregression model for multivariate time series analysis."""
    # Ensure the index is datetime and set daily frequency if not already defined
    if df.index.freq is None:
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('D')
    
    # Initialize and fit the VAR model
    model = VAR(df)
    return model.fit()

def fit_markov_switching(df):
    """Fit a Markov Switching model for regime changes in the data."""
    model = MarkovSwitching(df['Price'], k_regimes=2, switching_variance=True)
    return model.fit()

def fit_lstm(df, look_back=60):
    """Fit a Long Short-Term Memory (LSTM) model for time series forecasting."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))
    
    # Prepare the training data
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X, y, epochs=10, batch_size=32)
    return model, scaler
