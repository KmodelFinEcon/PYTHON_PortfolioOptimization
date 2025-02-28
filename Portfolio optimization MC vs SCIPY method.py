#min/max variance and volatility portfolio with VAR plot exercise (MC vs SCIPY Optimization compared) by KT.

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import scipy.optimize as sco
import matplotlib.pyplot as plt
from scipy import stats

#importing data from Yfinance

symbols = ['WNS', 'JD', 'VIPS', 'QSG', 'LOGI', 'GPN', 'CLVT', 'EEM']
start = datetime.datetime(2022, 12, 22)
end = datetime.datetime(2025, 2, 23)

try:
    df = yf.download(symbols, start, end, auto_adjust=False)['Adj Close']
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()
    
#global parameters    
 
portnum = 1000 #monte-carlo simulation
rfr = 0.0429 #risk free rate - treasury 10Y
days = 252
alpha = 0.05 # confidence interval 95%

log_returns = np.log(df / df.shift(1)).dropna() #log-returns
mean_returns = log_returns.mean()
cov = df.pct_change().dropna().cov() #covariance matrix construction

def portfoliop(weights, mean_returns, cov, rfr):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    sharpe_ratio = (portfolio_return - rfr) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio

# Simulate random portfolios
def RDMportfolios(portnum, mean_returns, cov, rfr):
    results_matrix = np.zeros((len(mean_returns) + 3, portnum))
    for i in range(portnum):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = portfoliop(weights, mean_returns, cov, rfr)
        results_matrix[:3, i] = [portfolio_return, portfolio_std, sharpe_ratio]
        results_matrix[3:, i] = weights
    results_df = pd.DataFrame(results_matrix.T, columns=['ret', 'stdev', 'sharpe'] + symbols)
    return results_df
results_frame = RDMportfolios(portnum, mean_returns, cov, rfr)

Sharpemaximump = results_frame.iloc[results_frame['sharpe'].idxmax()]
minimumvolp = results_frame.iloc[results_frame['stdev'].idxmin()]

plt.subplots(figsize=(20, 9))
plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, edgecolors="black", cmap='Grays')
plt.xlabel('Standard deviation')
plt.ylabel('Log Returns ')
plt.colorbar(label='SR')
plt.scatter(Sharpemaximump.iloc[1], Sharpemaximump.iloc[0], marker=(5, 2, 0), s=300, color='r', label='Max SR')
plt.scatter(minimumvolp.iloc[1], minimumvolp.iloc[0], marker=(5, 2, 0), s=300, color='c', label='Min Vol')
plt.title('Efficient Frontier MONTE CARLO METHOD')
plt.legend()
plt.show()

# Display optimal portfolios using monte-carlo method
print("\nMax Sharpe Ratio Portfolio:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(Sharpemaximump.to_frame().T)
print("\nMinimum Volatility Portfolio:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(minimumvolp.to_frame().T)

#SCIPY OPTIMIZATION ALGORITHM (SLSQP) 

def setup_optimization(num_assets):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))  # Weights between 0 and 1 
    initial_guess = np.array([1.0 / num_assets] * num_assets)
    return constraints, bounds, initial_guess

# Negative sharpe for minimization
def calc_neg_sharpe(weights, mean_returns, cov, rfr):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    sharpe_ratio = (portfolio_return - rfr) / portfolio_std
    return - sharpe_ratio 

# Maximize sharpe 
def max_sharpe_ratio(mean_returns, cov, rfr):
    num_assets = len(mean_returns)
    constraints, bounds, initial_guess = setup_optimization(num_assets)
    args = (mean_returns, cov, rfr)
    
    result1 = sco.minimize(calc_neg_sharpe, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    if not result1.success:
        raise ValueError("Optimization failed: " + result1.message)
    return result1

def calc_portfolio_std(weights, mean_returns, cov):
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    return portfolio_std

def min_variance(mean_returns, cov):
    num_assets = len(mean_returns)
    constraints, bounds, initial_guess = setup_optimization(num_assets)
    args = (mean_returns, cov)
    

    result = sco.minimize(calc_portfolio_std, initial_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    return result

if __name__ == "__main__":
    try:
        # Maximize Sharpe ratio
        optimal_port_sharpe = max_sharpe_ratio(mean_returns, cov, rfr)
        sharpe_weights = optimal_port_sharpe.x
        print("Optimal Weights for Maximum Sharpe Ratio:")
        print(pd.DataFrame([round(x, 2) for x in sharpe_weights], index=symbols, columns=['Weight']).T)
        
        # Minimize variance
        min_port_variance = min_variance(mean_returns, cov)
        min_var_weights = min_port_variance.x
        print("\nOptimal Weights for Minimum Variance:")
        print(pd.DataFrame([round(x, 2) for x in min_var_weights], index=symbols, columns=['Weight']).T)
    
    except ValueError as e:
        print(e)
        
def calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_VaR = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_return, portfolio_std, portfolio_VaR

def simulate_random_portfolios_VaR(portnum, mean_returns, cov, alpha, days):
    num_assets = len(mean_returns)
    results_matrix = np.zeros((num_assets + 3, portnum))
    
    for i in range(portnum):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, portfolio_VaR = calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha, days)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = portfolio_VaR
        results_matrix[3:, i] = weights 
    results_df = pd.DataFrame(results_matrix.T, columns=['ret', 'stdev', 'VaR'] + symbols)
    return results_df

results_frame1 = simulate_random_portfolios_VaR(portnum, mean_returns, cov, alpha, days)

min_VaR_port2 = results_frame1.iloc[results_frame1['VaR'].idxmin()]

plt.subplots(figsize=(20, 9))
plt.scatter(results_frame1['VaR'], results_frame1['ret'], c=results_frame1['VaR'], edgecolors="black", cmap='Spectral')
plt.xlabel('Value at Risk (VaR)')
plt.ylabel('Returns')
plt.colorbar(label='VaR')
plt.title('Efficient Frontier with Value at Risk')
plt.scatter(min_VaR_port2['VaR'], min_VaR_port2['ret'], marker=(5, 2, 0), color='b', s=300, label='Min VaR Portfolio')
plt.legend()
plt.show()

print("\nPortfolio with Minimum VaR:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(min_VaR_port2.to_frame().T)