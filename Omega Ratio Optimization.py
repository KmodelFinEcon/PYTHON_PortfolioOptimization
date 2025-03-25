# Sharpe, Omega ratio portfolio dynamics #
#   by k.tomov

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import quantstats as qs

# Data Fetch
tickers = ['JD', 'WNS', 'MSFT', 'SHOP', 'CRM', 'AMD', 'DIOD', 'INTC', 'QCOM', 'NVDA', 'XOM', 'WMB', 'EQT', 'SHEL', 'VIPS']
start = dt.date.today() - dt.timedelta(days=365*2)
end = dt.date.today() + dt.timedelta(days=1)

# Download data
returnstotal = yf.download(tickers, start, end)['Close']
returnstotal = returnstotal.dropna(axis=1, how='all')  # Drop tickers with all NaN
returnstotal.ffill(inplace=True)  # Forward fill remaining NaNs
tickers = returnstotal.columns.tolist()  # Update tickers list to valid columns

returns = returnstotal.pct_change().dropna()

# Calculate equal-weighted portfolio returns
equal_weights = np.ones(len(tickers)) / len(tickers)
portfolio_returns_equal = returns.dot(equal_weights)  # This creates a pandas Series

# Calculate portfolio metrics
portfolio_return = portfolio_returns_equal.mean() * 252
portfolio_volatility = portfolio_returns_equal.std() * np.sqrt(252)

# VaR and CVaR calculations
confidence_level = 0.95
var_95 = np.percentile(portfolio_returns_equal, 100 * (1 - confidence_level))
es_95 = portfolio_returns_equal.loc[portfolio_returns_equal <= var_95].mean()

print("Portfolio Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
print(f"Value at Risk (VaR) at 95% confidence level: {var_95:.2%}")
print(f"Expected Shortfall (ES) at 95% confidence level: {es_95:.2%}")

# Covariance matrix construction
mean_returns = returns.mean()
cov_matrix = returns.cov()

def sharpe_ratio(weights, returns):
    portfolio_returns = np.dot(returns, weights)
    portfolio_returns_series = pd.Series(portfolio_returns, index=returns.index)
    sharpe_ratio = qs.stats.sharpe(portfolio_returns_series, rf=0.054, periods=252, annualize=True)
    return -sharpe_ratio  # Negative to maximize

def omega_ratio(weights, returns):
    portfolio_returns = np.dot(returns, weights)
    portfolio_returns_series = pd.Series(portfolio_returns, index=returns.index)
    omega_ratio = qs.stats.omega(portfolio_returns_series.to_frame(), required_return=0, rf=0.054, periods=252)
    return -omega_ratio  # Negative to maximize

def cvar(weights, returns):
    portfolio_returns = np.dot(returns, weights)
    portfolio_returns_series = pd.Series(portfolio_returns, index=returns.index)
    cva = qs.stats.conditional_value_at_risk(portfolio_returns_series, sigma=1, confidence=0.95)
    return -cva  # Negative to maximize CVaR (minimize -CVaR)

constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
initial_guess = len(tickers) * [1. / len(tickers)]

# Optimize for maximum Sharpe ratio
result_sharpe = minimize(sharpe_ratio, initial_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

# Optimize for maximum Omega ratio
result_omega = minimize(omega_ratio, initial_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

# Optimize for minimum CVaR (corrected)
result_cva = minimize(cvar, initial_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

# Extract optimal weights
optimal_weights_sharpe = result_sharpe.x
optimal_weights_omega = result_omega.x
optimal_weights_cva = result_cva.x

def map_weights_to_tickers(weights, tickers):
    weighted_assets = list(zip(tickers, weights))
    weighted_assets.sort(key=lambda x: x[1], reverse=True)
    return weighted_assets

def print_optimal_weights(weights, tickers):
    weighted_assets = map_weights_to_tickers(weights, tickers)
    for ticker, weight in weighted_assets:
        print(f"{ticker}: {weight:.4f}")

df_optimal_weights = pd.DataFrame({
    'Sharpe': optimal_weights_sharpe,
    'Omega': optimal_weights_omega,
    'CVA': optimal_weights_cva
}, index=tickers)

print("\nOptimal Weights for Maximum Sharpe and Omega Ratios, and CVaR:")
print(df_optimal_weights)

# Calculate cumulative returns for each portfolio
optimal_portfolio_returns_sharpe = np.dot(returns, optimal_weights_sharpe)
optimal_portfolio_returns_omega = np.dot(returns, optimal_weights_omega)
optimal_portfolio_returns_cva = np.dot(returns, optimal_weights_cva)

cumulative_returns_sharpe = (1 + optimal_portfolio_returns_sharpe).cumprod()
cumulative_returns_omega = (1 + optimal_portfolio_returns_omega).cumprod()
cumulative_returns_cva = (1 + optimal_portfolio_returns_cva).cumprod()

# Plotting cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(cumulative_returns_sharpe, label='Maximum Sharpe Ratio Portfolio', color='blue')
plt.plot(cumulative_returns_omega, label='Maximum Omega Ratio Portfolio', color='green')
plt.plot(cumulative_returns_cva, label='CVaR-Optimized Portfolio', color='red')
plt.title('Cumulative Returns of Optimal Portfolios')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.tight_layout()
plt.show()

# Annualize metrics
def annualize_metrics(returns, trading_days=252):
    mean = np.mean(returns) * trading_days
    vol = np.std(returns) * np.sqrt(trading_days)
    return mean, vol

sharpe_mean, sharpe_vol = annualize_metrics(optimal_portfolio_returns_sharpe)
omega_mean, omega_vol = annualize_metrics(optimal_portfolio_returns_omega)
cva_mean, cva_vol = annualize_metrics(optimal_portfolio_returns_cva)

metrics_df = pd.DataFrame({
    'Sharpe': [sharpe_mean, sharpe_vol],
    'Omega': [omega_mean, omega_vol],
    'CVaR': [cva_mean, cva_vol]
}, index=['Annualized Return', 'Annualized Volatility'])

print("\nAnnualized Portfolio Metrics:")
print(metrics_df)

# Quantstats reports
print("\nQuantstats Report for Sharpe Portfolio:")
returns_sharpe = pd.Series(optimal_portfolio_returns_sharpe, index=returns.index)
qs.reports.basic(returns_sharpe, rf=0.0523)

print("\nQuantstats Report for Omega Portfolio:")
returns_omega = pd.Series(optimal_portfolio_returns_omega, index=returns.index)
qs.reports.basic(returns_omega, rf=0.0523)

print("\nQuantstats Report for CVaR Portfolio:")
returns_cva = pd.Series(optimal_portfolio_returns_cva, index=returns.index)
qs.reports.basic(returns_cva, rf=0.0523)