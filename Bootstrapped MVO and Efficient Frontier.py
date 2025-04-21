#Minimum Variance Boostrapped MVO with target yearly return Implementation by K.Tomov

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Global parameters 

n_bootstrap= 500 # Number of bootstrap samples
n_portfolios= 1000 # Portfolios per bootstrap
risk_free_rate= 0.047  # 1% annual risk-free 
target_return= 0.10 # 10% annual target

#data fetch

tickers= ['AMZN', 'AMD', 'AAPL', 'MSFT', 'LOGI']
startdate= '2021-01-01'
enddate= '2025-01-01'
data= yf.download(tickers, start=startdate, end=enddate)['Close']
returns = data.pct_change().dropna()

#Objective function and empty portfolio list

all_returns= []
all_risks= []
all_weights= []

#boostrap main computation 

for i in range(n_bootstrap):
    sample= returns.sample(n=len(returns), replace=True)
    mu= sample.mean() * 252 #trading days
    sigma= sample.cov() * 252 #trading days
    
    for i in range(n_portfolios):
        w= np.random.random(len(tickers))
        w/= w.sum()
        r= w.dot(mu)
        vol= np.sqrt(w.T.dot(sigma).dot(w))
        all_returns.append(r)
        all_risks.append(vol)
        all_weights.append(w)

all_returns= np.array(all_returns)
all_risks= np.array(all_risks)
all_weights= np.array(all_weights)

#Global constraint - target return

mask= all_returns >= target_return
if mask.any():
    idx_best= np.argmin(all_risks[mask])# Among those >= target, risk minimization is computed
    absolute_indices= np.where(mask)[0]
    best_idx= absolute_indices[idx_best]
    best_w= all_weights[best_idx]
    best_ret= all_returns[best_idx]
    best_vol= all_risks[best_idx]
else:
    best_w= best_ret = best_vol = None

#risk/return surface and efficient frontier

plt.figure(figsize=(12, 10))
sc= plt.scatter(all_risks, all_returns,c=(all_returns - risk_free_rate)/all_risks,cmap='magma', alpha=0.2)
plt.colorbar(sc, label='Sharpe Ratio')
plt.xlabel('Risk or Standard Deviation')
plt.ylabel('Annual Returns')
plt.title('Bootstrapped Portfolio Optimization')

# Efficient frontier from whole set
inds= np.argsort(all_risks)
risks_sorted= all_risks[inds]
returns_sorted= all_returns[inds]

eff_risks= []
eff_returns= []
current_max= -np.inf
for ret, vol in zip(returns_sorted, risks_sorted):
    if ret > current_max:
        current_max= ret
        eff_returns.append(ret)
        eff_risks.append(vol)

plt.plot(eff_risks, eff_returns,'r--',linewidth=2,label='Efficient Frontier')

#min risk at given target returns 

if best_w is not None:
    plt.scatter(best_vol, best_ret, c='red', s=100, edgecolor='k', label='Min-Risk at given Return')
    
plt.legend()
plt.grid(True)
plt.show()

#optimal portfolio

if best_w is not None:
    df_best = pd.DataFrame({'Ticker': tickers,'Weight': best_w}).set_index('Ticker')
    print(f"Minimum risk portfolio achieved > {target_return*100:.1f} percentage return:")
    print(df_best)
    print(f"\nExpected return: {best_ret:.2%}, Risk (sigma): {best_vol:.2%}")
else:
    print(f"no existant portfolio achieved target return {target_return*100:.1f}")
    
    