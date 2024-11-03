# src/eda.py
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_time_series(df):
    """Plot the Brent oil price over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Price'], label='Brent Oil Price')
    plt.title('Brent Oil Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def seasonal_decomposition(df):
    """Decompose the time series into trend, seasonality, and residuals."""
    decomposition = sm.tsa.seasonal_decompose(df['Price'], model='additive', period=365)
    decomposition.plot()
    plt.show()
