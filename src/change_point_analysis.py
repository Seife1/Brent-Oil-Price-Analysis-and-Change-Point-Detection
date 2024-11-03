# src/change_point_analysis.py
import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

def detect_change_points(df):
    """Detect change points in the time series data."""
    signal = df['Price'].values
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_points = algo.predict(pen=10)
    return change_points

def plot_change_points(df, change_points):
    """Plot change points on the time series data."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Price'], label='Brent Oil Price')
    for cp in change_points[:-1]:
        plt.axvline(df.index[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else "")
    plt.title('Brent Oil Price with Change Points')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
    
# use PyMC3 to set up a Bayesian model
def bayesian_change_point_analysis(df):
    """Perform Bayesian change point analysis on the time series data."""
    with pm.Model() as model:
        # Prior distribution for the change point
        change_point = pm.DiscreteUniform('change_point', lower=0, upper=len(df)-1)
        
        # Priors for the two normal distributions
        mu_1 = pm.Normal('mu_1', mu=df['Price'].mean(), sigma=df['Price'].std())
        mu_2 = pm.Normal('mu_2', mu=df['Price'].mean(), sigma=df['Price'].std())
        
        # Allocate appropriate mu based on the change point
        mu = pm.math.switch(change_point >= np.arange(len(df)), mu_1, mu_2)
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=df['Price'].std(), observed=df['Price'])
        
        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, cores=2)
        
    # Plot results
    pm.plot_posterior(trace)
    plt.show()    
    return trace

def plot_bayesian_change_points(df, trace):
    """Plot Bayesian change points on the time series data."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Price'], label='Brent Oil Price')
    plt.axvline(df.index[trace], color='red', linestyle='--', label='Change Point')
    plt.title('Brent Oil Price with Change Point')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
