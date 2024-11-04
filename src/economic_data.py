import wbdata
import pandas as pd

def fetch_economic_data(indicators, start_date, end_date):
    """Fetch economic data from World Bank for specified indicators and countries."""

    # Fetch the data
    data = wbdata.get_dataframe(indicators, country='all', date=(start_date, end_date), keep_levels=False)

    # Reset index for easier manipulation
    data.reset_index(inplace=True)

    # Step 1: Aggregate GDP by summing across countries for each year
    aggregated_gdp = data.groupby('date')['GDP (current US$)'].sum()

    # Step 2: Calculate Global Inflation Rate (weighted by GDP)
    data['Weighted Inflation'] = data['GDP (current US$)'] * data['Inflation, consumer prices (annual %)']
    aggregated_inflation = data.groupby('date').apply(lambda x: x['Weighted Inflation'].sum() / x['GDP (current US$)'].sum())

    # Step 3: Calculate Global Unemployment Rate (weighted by labor force)
    data['Weighted Unemployment'] = data['Total labor force'] * data['Unemployment, total (% of total labor force)']
    aggregated_unemployment = data.groupby('date').apply(lambda x: x['Weighted Unemployment'].sum() / x['Total labor force'].sum())

    # Step 4: Combine all aggregated data into a single DataFrame
    aggregated_data = pd.DataFrame({
    'date': aggregated_gdp.index,
    'Global GDP (current US$)': aggregated_gdp.values,
    'Global Inflation Rate (%)': aggregated_inflation.values,
    'Global Unemployment Rate (%)': aggregated_unemployment.values
    })
    
    return aggregated_data