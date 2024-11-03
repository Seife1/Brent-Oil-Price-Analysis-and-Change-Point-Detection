import pandas as pd

def load_data(filepath):
    """Load data from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=['Date'], dayfirst=True)
    df = df.dropna()  # Remove missing values
    df = df.sort_values('Date')  # Ensure data is sorted by date
    df.set_index('Date', inplace=True)
    return df