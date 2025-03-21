### Simple 1 Factor CAPM Model ###
#       by K.tomov

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Global Parameters
RISKFREE = 0.047
TRADING_DAYS = 252

def get_stock_data(tickers, start_date):
    data = yf.download(tickers, start=start_date)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data

def compute_simple_returns(data):
    return (data / data.shift(1) - 1).dropna()

def compute_beta(simple_returns, stock, market):
    cov_matrix = simple_returns.cov() * TRADING_DAYS
    beta = cov_matrix.loc[stock, market] / (simple_returns[market].var() * TRADING_DAYS)
    return beta

def compute_capm_return(simple_returns, stock, market, riskfree=RISKFREE, riskpremium='market'):
    if riskpremium == 'market':
        riskpremium = (simple_returns[market].mean() * TRADING_DAYS) - riskfree
    beta = compute_beta(simple_returns, stock, market)
    return riskfree + beta * riskpremium

def compute_sharpe(simple_returns, stock, capm_return, riskfree=RISKFREE):
    annual_std = simple_returns[stock].std() * np.sqrt(TRADING_DAYS)
    return (capm_return - riskfree) / annual_std

def plot_capm_line(simple_returns, stock, market):
    df = pd.DataFrame({
        "Market Return": simple_returns[market],
        "Stock Return": simple_returns[stock]
    }).dropna()
    
    slope, intercept = np.polyfit(df["Market Return"], df["Stock Return"], 1)  # Fit linear regression (CAPM)
    
    x_vals = np.linspace(df["Market Return"].min(), df["Market Return"].max(), 100)
    y_vals = slope * x_vals + intercept

    plt.figure(figsize=(10, 6))
    plt.scatter(df["Market Return"], df["Stock Return"], color='blue', label='Data points')
    plt.plot(x_vals, y_vals, color='red', linewidth=2, 
             label=f'Regression line: RA = {slope:.2f} RM + {intercept:.2f}')
    plt.title(f"CAPM Regression: {stock} vs {market}")
    plt.xlabel("Market Return (RM)")
    plt.ylabel("Stock Return (RA)")
    plt.legend()
    plt.grid(True)
    plt.show()

def stock_CAPM(stock_ticker, market_ticker, start_date='2015-01-01', riskfree=RISKFREE, riskpremium='market'):
    data = get_stock_data([stock_ticker, market_ticker], start_date)
    simple_returns = compute_simple_returns(data)
    beta = compute_beta(simple_returns, stock_ticker, market_ticker)
    capm_ret = compute_capm_return(simple_returns, stock_ticker, market_ticker, riskfree, riskpremium)
    sharpe = compute_sharpe(simple_returns, stock_ticker, capm_ret, riskfree)
    capmdata = pd.DataFrame({
        'Beta': [beta],
        'Return': [capm_ret],
        'Sharpe': [sharpe]
    }, index=[stock_ticker])
    return capmdata

tickers = ['F', 'PSCM', '^GSPC']
data = get_stock_data(tickers, start_date='2015-01-01')
print("Sample Data:")
print(data.head())

sec_returns = compute_simple_returns(data)
print("\nSimple Returns:")
print(sec_returns.head())

cov_matrix = sec_returns.cov() * TRADING_DAYS
print("\nAnnualized Covariance Matrix:")
print(cov_matrix)

beta_ford = compute_beta(sec_returns, 'F', '^GSPC')
beta_pscm = compute_beta(sec_returns, 'PSCM', '^GSPC')
print(f"\nBeta for Ford (F) vs. S&P500 (^GSPC): {beta_ford:.4f}")
print(f"Beta for PSCM vs. S&P500 (^GSPC): {beta_pscm:.4f}")

plot_capm_line(sec_returns, 'F', '^GSPC')

ford_capm = stock_CAPM("F", "^GSPC")
print("\nCAPM Metrics for Ford:")
print(ford_capm)

pscm_capm = stock_CAPM("PSCM", "^GSPC")
print("\nCAPM Metrics for PSCM:")
print(pscm_capm)

def omega_ratio(returns, required_return=0.0):
    # Convert annual required return to daily threshold
    daily_threshold = (1 + required_return) ** (1 / TRADING_DAYS) - 1
    excess_returns = returns - daily_threshold
    gains = excess_returns[excess_returns > 0].sum()
    losses = -excess_returns[excess_returns < 0].sum()
    if losses > 0:
        return gains / losses
    else:
        return np.nan

ford_returns = sec_returns['F']
PSCM_returns = sec_returns['PSCM']

print("\nSkewness of Ford's Returns:", ford_returns.skew())
print("Kurtosis of Ford's Returns:", ford_returns.kurtosis())
print("Omega Ratio for Ford:", omega_ratio(ford_returns, 0.05))
print("Omega Ratio for PSCM:", omega_ratio(PSCM_returns, 0.05))