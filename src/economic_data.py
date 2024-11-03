import wbdata
import datetime

def fetch_economic_data(indicators, countries, start_date, end_date):
    """Fetch economic data from World Bank for specified indicators and countries."""
    economic_data = wbdata.get_dataframe(indicators, country=countries, data_date=(start_date, end_date))
    return economic_data
