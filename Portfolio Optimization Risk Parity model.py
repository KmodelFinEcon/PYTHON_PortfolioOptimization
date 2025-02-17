import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
import seaborn as sns
import datetime
from pandas_datareader import data as web
import matplotlib.pyplot as plt

# Date range
start = '2020-01-01'
end = '2025-01-30'

# Tickers of assets
assets = ["VIPS", "WNS", "JD", "JPM", "MSFT", "LOGI", "CLVT", "GPN"]
assets.sort()  # Sort assets alphabetically

# Downloading data
data = yf.download(assets, start=start, end=end)
data = data.loc[:, 'Close']  # Extract only the 'Close' prices
data.columns = assets  # Set column names to asset tickers

# Calculating returns

def calculate_returns(data, return_type='simple'):
    
    if return_type == 'simple':
        returns = data.pct_change().dropna()# Calculate simple returns
    elif return_type == 'log':
        returns = np.log(data / data.shift(1)).dropna()# Calculate log returns
    else:
        raise ValueError("Invalid return_type. Choose 'simple' or 'log'.")
    
    return returns

return_type = 'log'  # Choose 'simple' or 'log'

Y = calculate_returns(data, return_type=return_type)

print(Y.head())

# Define the data source and the symbol for the risk-free rate
symbol = "TB3MS"

# Set the start and end dates for the data retrieval
end = datetime.datetime.now()
start = end - datetime.timedelta(days=365)  # Retrieve data for the past year

# DATA FROM FRED
try:
    data = web.DataReader(symbol, "fred", start, end)
    
    # Get the latest available annualized risk-free rate
    latest_rate = data.iloc[-1, 0]  # Extract the most recent value
    
    # Convert the rate to a float and print it
    risk_free_rate = float(latest_rate) / 100  # Convert percentage to decimal
    print(f"Latest Annualized US Risk-Free Rate: {risk_free_rate:.4f}")
    
except Exception as e:
    print(f"Error retrieving data: {e}")

# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Select method and estimate input parameters:
method_mu = 'hist'  # Method to estimate expected returns based on historical data
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data


# Estimate assets statistics
port.assets_stats(method_mu=method_mu, method_cov=method_cov)

# Estimate optimal portfolio:
model = 'Classic'  # Could be Classic (historical) or FM (Factor Model)
rm = 'MV'  # Risk measure used, this time will be variance
obj= 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = risk_free_rate  # Risk-free rate
b = None  # Risk contribution constraints vector
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

# Risk Parity optimization
w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)

# Plotting the composition of the portfolio
ax = rp.plot_pie(w=w_rp,
                 title='Risk Parity Variance',
                 others=0.05,  # Group assets with weights < 5% into "Others"
                 nrow=25,  # Maximum number of rows in the legend
                 cmap="tab20",  # Color map
                 height=6,  # Height of the figure
                 width=10,  # Width of the figure
                 ax=None)  # Axis to plot on (None creates a new figure)

# Show the plot

plt.show()

# Plotting the risk contribution per asset

mu = Y.mean()
cov = Y.cov() # Covariance matrix
returns = Y # Returns of the assets

ax = rp.plot_risk_con(w=w_rp,
                      cov=cov,
                      returns=returns,
                      rm=rm,
                      rf=rf,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      ax=None)

plt.show()

# Define risk measures
rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR', 'EVaR', 'CDaR', 'UCI', 'EDaR']

# Initialize DataFrame to store weights
w_s = pd.DataFrame([])

# Calculate optimal weights for each risk measure
for i in rms:
    w = port.rp_optimization(model='Classic', rm=i, rf=rf, b=None, hist=True)
    w_s = pd.concat([w_s, w], axis=1)

# Set column names to risk measures
w_s.columns = rms

# Display weights with background gradient (for Jupyter Notebook)
print(w_s.style.format("{:.2%}").background_gradient(cmap='YlGn'))

# Plotting the weights for each risk measure
plt.figure(figsize=(14, 8))
w_s.plot(kind='bar', stacked=True, colormap='tab20', rf=rf, figsize=(14, 8))
plt.title('Optimal Weights for Different Risk Measures')
plt.xlabel('Assets')
plt.ylabel('Weights')
plt.xticks(rotation=45)
plt.legend(title='Risk Measures', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Define risk measures
rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR', 'EVaR', 'CDaR', 'UCI', 'EDaR']

# Initialize DataFrame to store weights
w_s2 = pd.DataFrame([])

# Calculate optimal weights for each risk measure
for i in rms:
    w = port.rp_optimization(model='Classic', rm=i, rf=rf, obj=obj, b=None, hist=True)
    w_s2 = pd.concat([w_s2, w], axis=1)

# Set column names to risk measures
w_s2.columns = rms

# Convert weights to percentages
w_s2_percent = w_s2 * 100

# Plotting the heatmap using seaborn
plt.figure(figsize=(12, 6))
sns.heatmap(w_s2_percent, annot=True, fmt=".2f", cmap="YlGn", cbar=True, linewidths=0.5)

# Add labels and title
plt.title('Optimal Weights for Different Risk Measures (%)', fontsize=14)
plt.xlabel('Risk Measures', fontsize=12)
plt.ylabel('Assets', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()



                #####backtesting WIP #############



# Backtesting parameters
window = 252  # 1-year rolling window
rebalance_freq = 21  # Rebalance monthly (21 trading days)

# Initialize DataFrame to store portfolio returns
portfolio_returns = pd.DataFrame(index=Y.index[window:])

# Backtest Resampled Efficient Frontier
for i in range(window, len(Y), rebalance_freq):
    # Use rolling window of returns
    returns_window = Y.iloc[i-window:i]
    
    # Build portfolio object
    port = rp.Portfolio(returns=returns_window)
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    
    # Optimize portfolio
    weights = port.rp_optimization(model='Classic', rm='MV', rf=0, b=None, hist=True, resampled=True)
    
    # Calculate portfolio returns for the next period
    portfolio_returns.loc[Y.index[i], 'Resampled'] = np.dot(Y.iloc[i], weights)

# Backtest HRP
for i in range(window, len(Y), rebalance_freq):
    # Use rolling window of returns
    returns_window = Y.iloc[i-window:i]
    
    # Build portfolio object
    port = rp.Portfolio(returns=returns_window)
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    
    # Optimize portfolio
    weights = port.hrp_optimization(rm='MV', rf=0, b=None, hist=True)
    
    # Calculate portfolio returns for the next period
    portfolio_returns.loc[Y.index[i], 'HRP'] = np.dot(Y.iloc[i], weights)

# Drop NaN values
portfolio_returns.dropna(inplace=True)

# Plot cumulative returns
portfolio_returns.cumsum().plot(figsize=(10, 6), title='Cumulative Portfolio Returns')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.show()

#key perfromance metrics backtesting

# Annualized Return
annualized_return = portfolio_returns.mean() * 252

# Annualized Volatility
annualized_volatility = portfolio_returns.std() * np.sqrt(252)

# Sharpe Ratio (assuming risk-free rate = 0)
sharpe_ratio = annualized_return / annualized_volatility

# Maximum Drawdown
cumulative_returns = (1 + portfolio_returns).cumprod()
peak = cumulative_returns.cummax()
drawdown = (cumulative_returns - peak) / peak
max_drawdown = drawdown.min()

# Create a DataFrame of performance metrics
performance_metrics = pd.DataFrame({
    'Annualized Return': annualized_return,
    'Annualized Volatility': annualized_volatility,
    'Sharpe Ratio': sharpe_ratio,
    'Max Drawdown': max_drawdown
})

print(performance_metrics)