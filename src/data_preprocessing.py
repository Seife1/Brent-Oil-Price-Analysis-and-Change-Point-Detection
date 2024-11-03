import pandas as pd

def load_data(filepath):
    """Load data from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=['Date'], dayfirst=True)
    df = df.dropna()  # Remove missing values
    df = df.sort_values('Date')  # Ensure data is sorted by date
    df.set_index('Date', inplace=True)
    return df

def merge_economic_data(oil_data, economic_data):
    """Merge oil price data with economic indicators for multivariate analysis."""
    economic_data = economic_data.reset_index().pivot(index='date', columns='country')
    merged_data = oil_data.join(economic_data, on='Date', how='left')
    return merged_data
