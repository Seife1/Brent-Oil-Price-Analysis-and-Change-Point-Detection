from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def fit_var(df):
    model = VAR(df)
    return model.fit()

def fit_markov_switching(df):
    """Fit a Markov Switching model for regime changes in the data."""
    model = MarkovSwitching(df['Price'], k_regimes=2, trend='c', switching_variance=True)
    return model.fit()

def fit_lstm(df, look_back=60):
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
    
    model.fit(X, y, epochs=20, batch_size=32)
    return model, scaler
