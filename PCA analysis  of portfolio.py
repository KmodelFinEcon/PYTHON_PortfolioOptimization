#### Stock Portfolio PCA analysis againts various index exercise #####
#           by K.Tomov

import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
import pandas_datareader as web
import quantstats as qs
import seaborn as sns

# List of tickers:
tickers = ['JD', 'WNS', 'MSFT', 'SHOP','CRM', 'AMD', 'DIOD', 'INTC', 'QCOM', 'NVDA','XOM', 'WMB', 'EQT', 'SHEL', 'VIPS']

def get_stock_data(stocks, start_date, end_date):
    data = yf.download(stocks, start=start_date, end=end_date)['Close']
    return data

start_date = dt.datetime(2020, 12, 31)
start_date = start_date - dt.timedelta(days=2*365)
end_date = dt.datetime.today()

stock_data = get_stock_data(tickers, start_date, end_date)
returns = stock_data.pct_change().dropna() 

print("Stocks Daily Returns:")
print(returns.head())

# corporate bond yields data fetch
baa_yield = web.DataReader('BAA', 'fred', start_date, end_date).resample('D').interpolate(method='linear')
aaa_yield = web.DataReader('AAA', 'fred', start_date, end_date).resample('D').interpolate(method='linear')

# index data fetch
sp500 = yf.download('^GSPC', start=start_date, end=end_date)
dow = yf.download('^DJI', start=start_date, end=end_date)
nasdaq = yf.download('^IXIC', start=start_date, end=end_date)

# Calculate daily returns for indices
sp500_returns = sp500['Close'].pct_change().dropna()
dow_returns = dow['Close'].pct_change().dropna()
nasdaq_returns = nasdaq['Close'].pct_change().dropna()

def calculate_portfolio_returns(weights, returns):
    """Calculates cumulative portfolio returns from daily returns."""
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    return portfolio_returns, cumulative_returns

equal_weights = np.repeat(1/len(tickers), len(tickers))
portfolio_daily_returns, portfolio_cumulative_returns = calculate_portfolio_returns(equal_weights, returns)

##### PCA Analysis #####

pca = PCA()
principalComponents = pca.fit_transform(returns)
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

def create_component_variance_table(explained_variance):
    cumulative_variance = np.cumsum(explained_variance)
    component_variance_df = pd.DataFrame({
        'Component': range(1, len(explained_variance) + 1),
        'Explained Variance': explained_variance,
        'Cumulative Explained Variance': cumulative_variance
    })
    # For demonstration, concatenating first 7 and last 2 entries (adjust if needed)
    result_df = pd.concat([component_variance_df.head(7), component_variance_df.tail(2)])
    return result_df

def kaiser_rule_with_cutoff(explained_variance, cutoff=0.8):
    eigenvalues = explained_variance * len(explained_variance)
    kaiser_components = np.sum(eigenvalues > 1)
    cumulative_variance = np.cumsum(explained_variance)
    cutoff_components = np.argmax(cumulative_variance >= cutoff) + 1
    return min(kaiser_components, cutoff_components)

num_components = kaiser_rule_with_cutoff(explained_variance, cutoff=0.8)
print(f"Number of components based on Kaiser's rule with 80% cutoff: {num_components}")
component_variance_table = create_component_variance_table(explained_variance)
print("PCA Component Variance Table:\n", component_variance_table)

def find_outliers(data, threshold=3):
    """Finds outliers in the data using Z-score thresholding."""
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = data[z_scores > threshold]
    outliers = outliers.stack().reset_index(name='outlier')
    outliers.columns = ['date', 'ticker', 'outlier']
    return outliers

# Find outliers using a Z-score threshold of 3
outlier_data = find_outliers(returns.copy())
if not outlier_data.empty:
    print("Outliers detected (date, ticker, value):")
    print(outlier_data[['date','ticker', 'outlier']])
else:
    print("no outlier found")

# Align the portfolio and indices returns on the same dates
combined_returns = pd.DataFrame({
    'Portfolio': portfolio_daily_returns,
    'S&P500': sp500_returns,
    'Dow': dow_returns,
    'Nasdaq': nasdaq_returns
}).dropna()

# Calculate and plot the correlation matrix heatmap
correlation_matrix = combined_returns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap: Portfolio vs. Indices')
plt.show()

def calculate_risk_measures(daily_returns, label):
    sharpe_ratio = qs.stats.sharpe(daily_returns, rf=0.054, periods=252, annualize=True)
    omega_ratio = qs.stats.omega(daily_returns.to_frame(), required_return=0.5, rf=0.054, periods=252)
    cvar = qs.stats.conditional_value_at_risk(daily_returns, sigma=1, confidence=0.99)
    volatility = qs.stats.volatility(daily_returns)
    
    print(f"\n{label} Performance Ratios:")
    print(f"  Volatility: {volatility:.4f}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"  Omega Ratio: {omega_ratio:.4f}")
    print(f"  CVaR (99%): {cvar:.4f}\n")

# performance ratio
calculate_risk_measures(combined_returns['Portfolio'], "Portfolio")
calculate_risk_measures(combined_returns['S&P500'], "S&P500")