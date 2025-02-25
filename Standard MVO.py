####Mean Variance Optimization#### 
#       by K.Tomov
#assumptions: Risk Averse portfolio// zero constraint //stock returns are log-normal //Sequential Least Squares Programming method // weights between 0.5 and 25%

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import gmean


# Define the stock symbols and date range
symbols = ['JD', 'WNS', 'CRM', 'JPM', 'GPN']
start_date = '2021-01-01'
end_date = '2023-10-01'  # Use a date that is not in the future

# Download historical closing prices for the specified symbols and date range
data = yf.download(symbols, start=start_date, end=end_date)['Close']

# Plot the historical adjusted closing prices
data.plot(figsize=(10, 7))
plt.title('Historical Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend(title='Ticker')
plt.show()

# Compute daily percentage returns and drop missing values
returns = data.pct_change().dropna()

# Plot the daily returns
returns.plot(figsize=(10, 7))
plt.title('Daily Returns of Financial Assets')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.legend(title='Tickers')
plt.show()

# Sharpe based objective function
def objective_sharpe(weights): 
    return -np.dot(weights, returns.mean()) / np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

cov_matrix = returns.cov() * 252  # 252 trading days in a year
print("Covariance Matrix:\n", cov_matrix)

# CVAR based objective function
def objective_cvar(weights):
    portfolio_returns = np.dot(returns, weights)
    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()
    conf_level = 0.05  # Confidence interval
    cvar = portfolio_mean - portfolio_std * norm.ppf(conf_level)
    return cvar

# SORTINO Objective function (#maximization of SR)
def objective_sortino(weights):
    portfolio_returns = np.dot(returns, weights)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = portfolio_returns.mean() / downside_std
    return - sortino_ratio  

# Minimization of variance
def objective_variance(weights):
    return np.dot(weights.T, np.dot(returns.cov() * 252, weights))

# Constraints (ensuring all weight bounds = 100%)
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

#weights bounds (NULL) -> CAN be added
bounds = tuple((0.05, 0.25) for _ in range(len(symbols))) #portfolio bounds

optimization_criterion = 'cvar'  # Choose from below

# Computation
init_guess = np.array(len(symbols) * [1. / len(symbols),])
if optimization_criterion == 'sharpe':
    opt_results = minimize(objective_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
elif optimization_criterion == 'cvar':
    opt_results = minimize(objective_cvar, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
elif optimization_criterion == 'sortino':
    opt_results = minimize(objective_sortino, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
elif optimization_criterion == 'variance':
    opt_results = minimize(objective_variance, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

# Weights Output
optimal_weights = opt_results.x


# Optimization of all Ratios
opt_results_cvar = minimize(objective_cvar, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
opt_results_sortino = minimize(objective_sortino, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
opt_results_variance = minimize(objective_variance, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
opt_results_sharpe = minimize(objective_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

# Optimal weights for each criterion
optimal_weights_cvar = opt_results_cvar.x
optimal_weights_sortino = opt_results_sortino.x
optimal_weights_variance = opt_results_variance.x
optimal_weights_sharpe = opt_results_sharpe.x

#efficient frontier plot
port_returns = []
port_volatility = []
sharpe_ratio = []
all_weights = []  #simulated weights

num_assets = len(symbols)
num_portfolios = 4000

np.random.seed(101)

for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns_portfolio = np.dot(weights, returns.mean()) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sr = returns_portfolio / volatility
    sharpe_ratio.append(sr)
    port_returns.append(returns_portfolio)
    port_volatility.append(volatility)
    all_weights.append(weights)  # Recording weights

plt.figure(figsize=(12, 8))
plt.scatter(port_volatility, port_returns, c=sharpe_ratio, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

#Calculate and plot the returns and volatility of the optimal portfolio for each criterion.

opt_returns_cvar = np.dot(optimal_weights_cvar, returns.mean()) * 252
opt_volatility_cvar = np.sqrt(np.dot(optimal_weights_cvar.T, np.dot(returns.cov() * 252, optimal_weights_cvar)))
opt_portfolio_cvar = plt.scatter(opt_volatility_cvar, opt_returns_cvar, color='hotpink', s=50, label='CVaR')

opt_returns_sortino = np.dot(optimal_weights_sortino, returns.mean()) * 252
opt_volatility_sortino = np.sqrt(np.dot(optimal_weights_sortino.T, np.dot(returns.cov() * 252, optimal_weights_sortino)))
opt_portfolio_sortino = plt.scatter(opt_volatility_sortino, opt_returns_sortino, color='g', s=50, label='Sortino')

opt_returns_variance = np.dot(optimal_weights_variance, returns.mean()) * 252
opt_volatility_variance = np.sqrt(np.dot(optimal_weights_variance.T, np.dot(returns.cov() * 252, optimal_weights_variance)))
opt_portfolio_variance = plt.scatter(opt_volatility_variance, opt_returns_variance, color='b', s=50, label='Variance')

opt_returns_sharpe = np.dot(optimal_weights_sharpe, returns.mean()) * 252
opt_volatility_sharpe = np.sqrt(np.dot(optimal_weights_sharpe.T, np.dot(returns.cov() * 252, optimal_weights_sharpe)))
opt_portfolio_sharpe = plt.scatter(opt_volatility_sharpe, opt_returns_sharpe, color='r', s=50, label='Sharpe')

plt.legend(loc='upper right')
plt.show()


# Maximum drawdown
def max_drawdown(return_series):
    comp_ret = (1 + return_series).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak) - 1
    return dd.min()

# Descriptive statistics
def detailed_portfolio_statistics(weights):
    portfolio_returns = returns.dot(weights)
    
    # General stats
    mean_return_annualized = gmean(portfolio_returns + 1)**252 - 1
    std_dev_annualized = portfolio_returns.std() * np.sqrt(252)
    skewness = skew(portfolio_returns)
    kurt = kurtosis(portfolio_returns)
    max_dd = max_drawdown(portfolio_returns)
    count = len(portfolio_returns)
    
    # OPtimization metrics
    risk_free_rate = 0.04
    sharpe_ratio = (mean_return_annualized - risk_free_rate) / std_dev_annualized
    conf_level = 0.05
    cvar = mean_return_annualized - std_dev_annualized * norm.ppf(conf_level)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std_dev = downside_returns.std() * np.sqrt(252)
    sortino_ratio = mean_return_annualized / downside_std_dev
    variance = std_dev_annualized ** 2 
    
    return mean_return_annualized, std_dev_annualized, skewness, kurt, max_dd, count, sharpe_ratio, cvar, sortino_ratio, variance

# Descriptive statistics for individual portfolios
statistics_cvar = detailed_portfolio_statistics(optimal_weights_cvar)
statistics_sortino = detailed_portfolio_statistics(optimal_weights_sortino)
statistics_variance = detailed_portfolio_statistics(optimal_weights_variance)
statistics_sharpe = detailed_portfolio_statistics(optimal_weights_sharpe)

# number of stats
statistics_names = ['Anualized returns', 'Volatility Anualized', 'Skewness', 'Kurtosis', 'Max Drawdown', 'data count', 'Sharpe Ratio', 'CVaR', 'Sortino ratio', 'Variance']

# Dictionary that associates the names of the optimization methods with the optimal weights and statistics.
portfolio_data = {
    'CVaR': {
        'weights': optimal_weights_cvar,
        'statistics': detailed_portfolio_statistics(optimal_weights_cvar)
    },
    'Sortino': {
        'weights': optimal_weights_sortino,
        'statistics': detailed_portfolio_statistics(optimal_weights_sortino)
    },
    'Variance': {
        'weights': optimal_weights_variance,
        'statistics': detailed_portfolio_statistics(optimal_weights_variance)
    },
    'Sharpe': {
        'weights': optimal_weights_sharpe,
        'statistics': detailed_portfolio_statistics(optimal_weights_sharpe)
    },
}

# print output
for method, data in portfolio_data.items():
    print("\n")
    print("--------Output1----------")
    print("\n")
    print(f"optimal portfolio weight for {method}:")
    print("\n")
    for symbol, weight in zip(symbols, data['weights']):
        if weight < 1e-4:  # any weight bellow 0.01% are NULL
            print(f"{symbol}: is NULL ")
        else:
            print(f"{symbol}: {weight*100:.2f}%")

    print("\n")
    print(f"Descriptive statistics of the optimal portfolio for {method}:")
    print("\n")
    for name, stat in zip(statistics_names, data['statistics']):
        print(f"{name}: {stat*100 if name != 'data count' else stat:.2f}")

print("\n")
print("--------Output2--------")
