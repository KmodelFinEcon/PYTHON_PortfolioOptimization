
#Portfolio ratio computation [5 year lookback] by K.Tomov

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

#data fetch

enddate = dt.datetime.today()
start = enddate - dt.timedelta(days=5 * 365)
tickers = ['WNS', 'JD', 'VIPS', 'BIDU']
data = yf.download(tickers, start=start, end=enddate)['Close']
rfr = 0.0446
tradingdays = 255

#log returns
log_returns = np.log(data / data.shift(1)).dropna()
log_returns['Port'] = log_returns.mean(axis=1)
cumulative_returns = np.exp(log_returns.cumsum())
cumulative_returns.plot(figsize=(7, 3), title="Cumulative Log Returns")
plt.tight_layout()
plt.show()

# Risk metric functions
def sharpe_ratio(series, N=tradingdays, rf=rfr):
    excess_return = series.mean() * N - rf
    risk = series.std() * np.sqrt(N)
    return excess_return / risk

def sortino_ratio(series, N=tradingdays, rf=rfr):
    excess_return = series.mean() * N - rf
    downside_std = series[series < 0].std() * np.sqrt(N)
    return excess_return / downside_std if downside_std != 0 else np.nan

def max_drawdown(series):
    comp_ret = np.exp(series.cumsum())
    peak = comp_ret.cummax()
    drawdown = comp_ret / peak - 1
    return drawdown.min()

def calmar_ratio(series, N=tradingdays):
    return series.mean() * N / abs(max_drawdown(series)) if max_drawdown(series) != 0 else np.nan

yearly_stats = {}
years = log_returns.index.year.unique()
rf = rfr
N = tradingdays

#statistics of portfolio ratios

for year in sorted(years)[-5:]:  #5 represents years of ratio computation
    yearly_data = log_returns[log_returns.index.year == year]
    stats = pd.DataFrame({
        'Sharpe Ratio': yearly_data.apply(sharpe_ratio, args=(N, rf)),
        'Sortino Ratio': yearly_data.apply(sortino_ratio, args=(N, rf)),
        'Max Drawdown': yearly_data.apply(max_drawdown),
        'Calmar Ratio': yearly_data.apply(calmar_ratio)
    })
    stats['Year'] = year
    yearly_stats[year] = stats

#single data frame construction 
all_stats = pd.concat(yearly_stats.values())
all_stats = all_stats.set_index(['Year'], append=True).reorder_levels(['Year', all_stats.index.names[0]])
print("\nRisk Ratios for Last 5 Years [annualized]:\n")
print(all_stats.round(4))

# Recent portfolio statistics
full_stats = pd.DataFrame({
    'Sharpe Ratio': log_returns.apply(sharpe_ratio, args=(N, rf)),
    'Sortino Ratio': log_returns.apply(sortino_ratio, args=(N, rf)),
    'Max Drawdown': log_returns.apply(max_drawdown),
    'Calmar Ratio': log_returns.apply(calmar_ratio)
})

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
full_stats['Sharpe Ratio'].plot(kind='bar', ax=axs[0, 0], title='Sharpe Ratio')
full_stats['Sortino Ratio'].plot(kind='bar', ax=axs[0, 1], title='Sortino Ratio')
full_stats['Max Drawdown'].plot(kind='bar', ax=axs[1, 0], title='Max Drawdown')
full_stats['Calmar Ratio'].plot(kind='bar', ax=axs[1, 1], title='Calmar Ratio')

for ax in axs.flat:
    ax.set_ylabel('Value')
    ax.set_xlabel('Asset')

plt.tight_layout()
plt.show()

#cumulative returns chart
fig, ax = plt.subplots(figsize=(10, 6))
cumulative_returns.plot(ax=ax)
table_data = np.round(full_stats.values, 2)
plt.table(cellText=table_data,
          colLabels=full_stats.columns,
          rowLabels=full_stats.index,
          loc='top',
          cellLoc='center',
          rowLoc='center',
          colWidths=[0.2] * len(full_stats.columns))
plt.tight_layout()
plt.show()
