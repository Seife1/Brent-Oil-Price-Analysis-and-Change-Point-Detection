from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,r2_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def fit_var(df):
    """Fit a Vector Autoregression (VAR) model for multivariate time series analysis."""
    # Ensure the index is datetime and set daily frequency if not already defined
    if df.index.freq is None:
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('D')
    
    # Initialize and fit the VAR model
    model = VAR(df)
    var_model = model.fit()
    
    return var_model

def evaluate_var_model(var_model, df, steps=5):
    """Evaluate VAR model by calculating forecast errors and plotting actual vs predicted values."""
    forecast = var_model.forecast(y=df.values[-var_model.k_ar:], steps=steps)
    forecast_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)
    
    # Evaluate by calculating MAE, MSE, RMSE
    print("VAR Model Evaluation Metrics:")
    for col in df.columns:
        actual = df[col].values[-steps:]
        predicted = forecast_df[col].values
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        print(f"{col} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    
    # Plot the actual vs predicted values
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=f'Actual {col}')
        plt.plot(forecast_df.index, forecast_df[col], linestyle='--', label=f'Forecasted {col}')
    plt.title('VAR Model Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def fit_markov_switching(df):
    """Fit a Markov Switching Autoregression model for detecting regime changes."""
    model = MarkovAutoregression(df['Price'], k_regimes=2, order=4, switching_ar=False)
    markov_model = model.fit()
    
    return markov_model

def evaluate_markov_switching_model(markov_model, df):
    """Evaluate and plot the Markov Switching model for regime changes."""
    # Predicted regimes
    df['Predicted Regime'] = markov_model.predict()
    
    # Plot the Price with Regimes
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Price'], label='Price')
    plt.scatter(df.index, df['Price'], c=df['Predicted Regime'], cmap='viridis', label='Regime')
    plt.title('Price with Markov Switching Regimes')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Align y_true and y_pred to the same length
    y_true = df['Price'].iloc[:len(markov_model.fittedvalues)]
    y_pred = markov_model.fittedvalues
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print("Markov Switching Model Evaluation Metrics:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)



def fit_lstm(df, look_back=60, model_path='../dashboard/backend/models/lstm_model.h5', scaler_path='../dashboard/backend/models/scaler.pkl'):
    """
    Fit a Long Short-Term Memory (LSTM) model for time series forecasting, 
    and save both the model and the scaler.
    """
    
    # Initialize and fit the MinMaxScaler on the 'Price' column
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))
    
    # Prepare the training data with a specified look-back period
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape input to be [samples, time steps, features] for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Define the LSTM model architecture
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile the model with optimizer and loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model on the prepared dataset
    model.fit(X, y, epochs=20, batch_size=32)
    
    # Save the trained model as an .h5 file
    model.save(model_path)
    
    # Save the fitted scaler as a .pkl file for later use
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    print(f"Model saved as {model_path} and scaler saved as {scaler_path}")

def evaluate_lstm_model(y_true, y_pred):
    """
    Evaluate the model's performance by calculating common regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2_score_value = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Print evaluation metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-Squared (R2 Score):", r2_score_value)
    print("Mean Absolute Percentage Error (MAPE):", mape * 100, "%")

def plot_result(y_test, y_pred):
    """
    Plot the actual versus predicted values to visually assess model performance.
    """
    # Flatten y_test and y_pred to 1-dimensional arrays
    actual_vs_prediction = pd.DataFrame({
        "Original Price": y_test.ravel(),
        "Predicted Price": y_pred.ravel()
    })
    
    # Plotting actual vs predicted prices
    plt.figure(figsize=(14, 5))
    plt.plot(actual_vs_prediction.index, actual_vs_prediction['Original Price'], label="Original Price")
    plt.plot(actual_vs_prediction.index, actual_vs_prediction['Predicted Price'], label="Predicted Price", linestyle='--')
    plt.title('LSTM Model Brent Oil Price Prediction')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
