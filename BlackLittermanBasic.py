#Data source
import yfinance as yf

#normal imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Use PyPortfolioOpt for Calculations
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import DiscreteAllocation

#Create a Portfolio
symbols = [
    'VIPS',
    'WNS',
    'JD',
    'JPM',
    'MSFT',
    'LOGI',
    'CLVT',
    'GPN',
 
]

#Get the stock data
portfolio = yf.download(symbols, start="2018-01-01", end="2025-02-28")['Adj Close']

portfolio.head()

#SP500 ETF Benchmark
market_prices = yf.download("SPY", start='2018-01-01', end='2025-02-28')["Adj Close"]
market_prices.head()

#Grap Market Capitalization for each stock in portfolio
mcaps = {}
for t in symbols:
    stock = yf.Ticker(t)
    mcaps[t] = stock.info["marketCap"]
mcaps

"""## Step 2: Getting Priors"""

#Calculate Sigma and Delta to get implied market returns
#Ledoit-Wolf is a particular form of shrinkage, where the shrinkage coefficient is computed using O?
S = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()

delta = black_litterman.market_implied_risk_aversion(market_prices)
delta

#Visualize the Covariant Correlation
sns.heatmap(S.corr(), cmap='coolwarm')

market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
market_prior

#What am I looking at here?
market_prior.plot.barh(figsize=(10,5));

"""## Step 3: Integrating Views"""

#You don't have to provide views on all the assets
viewdict = {
    'VIPS',
    'WNS',
    'JD',
    'JPM',
    'MSFT',
    'LOGI',
    'CLVT',
    'GPN',
}

bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)

"""### Creating Confidences

we can calculate the uncertainty matrix directly by specifying 1 standard deviation confidence intervals, i.e bounds which we think will contain the true return 68% of the time. This may be easier than coming up with somewhat arbitrary percentage confidences
"""

intervals = [
    (0, 0.25),
    (0.1, 0.4),
    (-0.1, 0.15),
    (-0.05, 0.1),
    (0.15, 0.25),
    (-0.1, 0),
    (0.1, 0.2),
    (0.08, 0.12),

]

variances = []
for lb, ub in intervals:
    sigma = (ub - lb)/2
    variances.append(sigma ** 2)

print(variances)
omega = np.diag(variances)

"""# Step 4: Calculate Posterior Estimate Returns"""

fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(omega)

# We want to show all ticks...
ax.set_xticks(np.arange(len(bl.tickers)))
ax.set_yticks(np.arange(len(bl.tickers)))

ax.set_xticklabels(bl.tickers)
ax.set_yticklabels(bl.tickers)
plt.show()

# We are using the shortcut to automatically compute market-implied prior
bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                        absolute_views=viewdict, omega=omega)

# Posterior estimate of returns
ret_bl = bl.bl_returns()
ret_bl

rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)],
             index=["Prior", "Posterior", "Views"]).T
rets_df

rets_df.plot.bar(figsize=(12,8));

S_bl = bl.bl_cov()
plotting.plot_covariance(S_bl);

"""# Step 5: Portfolio Allocation"""

ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()
weights

pd.Series(weights).plot.pie(figsize=(9,9));

from pypfopt.plotting import plot_weights

# Maximum Sharpe
ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()

plot_weights(weights)
ef.portfolio_performance(verbose = True, risk_free_rate = 0.009)