# src/time_series_models.py
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt

def fit_arima_model(df, order=(1, 1, 1)):
    """Fit an ARIMA model to the data."""
    model = ARIMA(df['Price'], order=order)
    arima_result = model.fit()
    return arima_result

def plot_arima_forecast(df, arima_result, steps=30):
    """Plot forecast from ARIMA model."""
    forecast = arima_result.get_forecast(steps=steps)
    forecast_ci = forecast.conf_int()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Price'], label='Observed')
    plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def fit_garch_model(df):
    """Fit a GARCH model to the data."""
    garch_model = arch_model(df['Price'], vol='Garch', p=1, q=1)
    garch_result = garch_model.fit(disp="off")
    return garch_result

def plot_garch_volatility(garch_result):
    """Plot conditional volatility from GARCH model."""
    plt.figure(figsize=(10, 6))
    plt.plot(garch_result.conditional_volatility)
    plt.title('Conditional Volatility from GARCH Model')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.show()
