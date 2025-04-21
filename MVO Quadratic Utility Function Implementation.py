# Optimal Portfolio derived from Quadratic Utility function with Exogenous constraints with volatility calibrated risk aversion estimation [SCYPY VS CVXPY]
# By K.Tomov

import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

TICKERS = ['JD', 'WNS', 'BIDU', 'LOGI']
START_DATE = '2020-01-01'
END_DATE = '2023-04-18'
TARGET_VOL = 0.08 # Target annualized volatility for gamma calibration (Can be assumed or derived from GARCH model)
R_TARGET = 0.11 # Yearly target return constraint
GAMMA_INITIAL = 3.0 # Initial guess of GAMMA

# Bounds on weights -> first asset has [0.05,0.4], others [0.1,0.4]

def get_bounds(n):
    return [(0.05, 0.4) if i == 0 else (0.1, 0.4) for i in range(n)]
SUM_CONSTR = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

#data fetch function

def fetch_price_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)['Close']
    return df.pct_change().dropna()

def annualize_stats(rets, periods=252): # trading days
    mu = rets.mean().values * periods
    Sigma = rets.cov().values * periods
    return mu, Sigma

#objective optimization function 

# CVXPY utility (no return constraint)
def optimize_utility(mu, Sigma, gamma, bounds):
    n = len(mu)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1,
                   w >= [b[0] for b in bounds],
                   w <= [b[1] for b in bounds]]
    utility = w @ mu - (gamma / 2) * cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Maximize(utility), constraints)
    prob.solve(warm_start=True)
    return w.value

# CVXPY utility with return constraint
def optimize_quadratic_utility(mu, Sigma, gamma, r_target, bounds):
    n = len(mu)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1,
                   w >= [b[0] for b in bounds],
                   w <= [b[1] for b in bounds],
                   w @ mu >= r_target]
    utility = w @ mu - (gamma / 2) * cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Maximize(utility), constraints)
    prob.solve(warm_start=True)
    return w.value, prob.status

# CVXPY max-return and min-variance

def optimize_max_return(mu, bounds):
    n = len(mu)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1,
                   w >= [b[0] for b in bounds],
                   w <= [b[1] for b in bounds]]
    prob = cp.Problem(cp.Maximize(w @ mu), constraints)
    prob.solve(warm_start=True)
    return w.value, prob.status

def optimize_min_variance(Sigma, bounds):
    n = Sigma.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1,
                   w >= [b[0] for b in bounds],
                   w <= [b[1] for b in bounds]]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    prob.solve(warm_start=True)
    return w.value, prob.status

#risk aversion (GAMMA CALIBRATION)

def portfolio_vol(weights, Sigma):
    return np.sqrt(weights @ Sigma @ weights)

def calibrate_gamma(mu, Sigma, bounds, target_vol, gammas=None): #grid serach for target parameters
    if gammas is None:
        gammas = np.logspace(-1, 2, 50)  # from 0.1 to 100
    vols = []
    for G in gammas:
        w = optimize_utility(mu, Sigma, G, bounds)
        vols.append(portfolio_vol(w, Sigma))
    idx = np.argmin(np.abs(np.array(vols) - target_vol))
    return gammas[idx]

#scipy optimization 

def objective_scipy(w, mu, Sigma, gamma):
    return 0.5 * gamma * w @ Sigma @ w - mu @ w


def optimize_scipy(mu, Sigma, gamma, r_target, bounds):
    cons = [SUM_CONSTR,
            {'type': 'ineq', 'fun': lambda w: w @ mu - r_target}]
    w0 = np.ones(len(mu)) / len(mu)
    res = minimize(objective_scipy, w0,
                   args=(mu, Sigma, gamma),
                   bounds=bounds,
                   constraints=cons,
                   options={'disp': False})
    return (res.x, res)

#efficient frontier

def compute_efficient_frontier(mu, Sigma, bounds, n_points=50):
    n = len(mu)
    w = cp.Variable(n)
    ret_param = cp.Parameter(nonneg=True)
    constraints = [cp.sum(w) == 1,# Fully invested
                   w >= [b[0] for b in bounds], # more or equal ...
                   w <= [b[1] for b in bounds], #less or equal ..
                   w @ mu >= ret_param] # Return ≥ 11%
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)

    w_max, _ = optimize_max_return(mu, bounds)
    w_min, _ = optimize_min_variance(Sigma, bounds)
    rets = np.linspace(mu @ w_min, mu @ w_max, n_points)

    vols = []
    eff_rets = []
    for R in rets:
        ret_param.value = R
        prob.solve(warm_start=True)
        if prob.status == 'optimal':
            wv = w.value
            eff_rets.append(mu @ wv)
            vols.append(portfolio_vol(wv, Sigma))
    return np.array(vols), np.array(eff_rets)

def plot_frontier(vols, rets, mu, Sigma, weights_cvx, weights_scipy, tickers):
    plt.figure(figsize=(10,6))
    plt.plot(vols, rets, label='Efficient Frontier')
    sigma_i = np.sqrt(np.diag(Sigma))
    plt.scatter(sigma_i, mu, marker='o', label='Assets')
    for i, t in enumerate(tickers): plt.annotate(t, (sigma_i[i], mu[i]))
    if weights_cvx is not None:
        vc = portfolio_vol(weights_cvx, Sigma)
        rc = mu @ weights_cvx
        plt.scatter(vc, rc, marker='*', s=200, label='CVXPY')
    if weights_scipy is not None:
        vs = portfolio_vol(weights_scipy, Sigma)
        rs = mu @ weights_scipy
        plt.scatter(vs, rs, marker='*', s=200, label='SciPy')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title('Efficient Frontier with Calibrated Γ')
    plt.legend(); plt.grid(True)
    plt.show()

#Code execusion

def main():
    rets = fetch_price_data(TICKERS, START_DATE, END_DATE)
    mu, Sigma = annualize_stats(rets)
    bounds = get_bounds(len(mu))

    gamma_calibrated = calibrate_gamma(mu, Sigma, bounds, TARGET_VOL)
    print(f"Calibrated r = {gamma_calibrated:.3f} to target vol {TARGET_VOL:.2%}")

    #optimizers compared

    w_cvx, status_cvx = optimize_quadratic_utility(mu, Sigma, gamma_calibrated, R_TARGET, bounds)
    w_sci, res_sci = optimize_scipy(mu, Sigma, gamma_calibrated, R_TARGET, bounds)

    vols, rets_f = compute_efficient_frontier(mu, Sigma, bounds)
    plot_frontier(vols, rets_f, mu, Sigma,
                  w_cvx if status_cvx=='optimal' else None,
                  w_sci if res_sci.success else None,
                  TICKERS)

    print("\nCVXPY with Calibrated r= ")
    print("Status:", status_cvx)
    print("Weights:", w_cvx)
    print("Return:", np.round(mu @ w_cvx,4))
    print("Volatility:", np.round(portfolio_vol(w_cvx, Sigma),4))

    print("\nSciPy with Calibrated r=")
    print("Success:", res_sci.success)
    print("Weights:", w_sci)
    print("Return:", np.round(mu @ w_sci,4))
    print("Volatility:", np.round(portfolio_vol(w_sci, Sigma),4))

if __name__ == "__main__":
    main()
