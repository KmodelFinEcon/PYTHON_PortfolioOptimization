#Portfolio Optimization via Pair Copula-GARCH-EVT-CVaR Model Implementation by K.Tomov from 2011 by Ling Deng, Chaoqun Ma, Wenyu Yang

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from scipy.stats import genpareto, norm
from statsmodels.distributions.copula.api import GaussianCopula
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime

#Get return function for main execusion:

def load_returns(symbols, start_date, end_date):
    prices = yf.download(symbols, start=start_date, end=end_date)['Close']
    returns = prices.pct_change().dropna()
    return returns

def fit_gjr_garch(returns):
    garch_results = {}
    sigma_last = {}
    eps_std = pd.DataFrame(index=returns.index)

    for asset in returns.columns:
        am = arch_model(returns[asset] * 100, vol='GARCH', p=1, o=1, q=1, dist='normal')
        res = am.fit(disp='off')
        garch_results[asset] = res
        eps_std[asset] = res.std_resid
        sigma_last[asset] = res.conditional_volatility.iloc[-1]

    return garch_results, eps_std, sigma_last

#pareto tail fit uysing empirical CDF, returning EVY parameters
def fit_evt(eps_std):
    ev_params = {}
    empirical_cdfs = {}

    for asset in eps_std.columns:
        z = eps_std[asset].dropna().values
        uL, uR = np.percentile(z, 10), np.percentile(z, 90)
        data_l = -(z[z < uL] - uL)
        data_r = (z[z > uR] - uR)

        c_l, loc_l, scale_l = genpareto.fit(data_l)
        c_r, loc_r, scale_r = genpareto.fit(data_r)

        ev_params[asset] = {
            'uL': uL, 'uR': uR,
            'c_l': c_l, 'loc_l': loc_l, 'scale_l': scale_l,
            'c_r': c_r, 'loc_r': loc_r, 'scale_r': scale_r,
            'n': len(z), 'n_l': len(data_l), 'n_r': len(data_r)
        }

        sorted_z = np.sort(z)
        ecdf_vals = np.arange(1, len(z) + 1) / len(z)
        empirical_cdfs[asset] = (sorted_z, ecdf_vals)

    return ev_params, empirical_cdfs

def marginal_cdf(x, params, sorted_z, ecdf_vals):
    p_l = params['n_l'] / params['n']
    p_r = params['n_r'] / params['n']
    uL, uR = params['uL'], params['uR']

    if x < uL:
        return p_l * genpareto.cdf(-(x - uL), params['c_l'], loc=params['loc_l'], scale=params['scale_l'])
    elif x > uR:
        return 1 - p_r * genpareto.sf(x - uR, params['c_r'], loc=params['loc_r'], scale=params['scale_r'])
    else:
        return p_l + np.interp(x, sorted_z, ecdf_vals) * (1 - p_l - p_r)


def transform_to_uniforms(eps_std, ev_params, empirical_cdfs):
    uniforms = eps_std.copy(deep=True)
    for asset in eps_std.columns:
        params = ev_params[asset]
        sorted_z, ecdf_vals = empirical_cdfs[asset]
        uniforms[asset] = eps_std[asset].apply(lambda x: marginal_cdf(x, params, sorted_z, ecdf_vals))
    return uniforms.values

def fit_gaussian_copula(uniforms):
    norm_scores = norm.ppf(uniforms)
    emp_corr = np.corrcoef(norm_scores, rowvar=False)
    n_dim = uniforms.shape[1]
    gc = GaussianCopula(corr=emp_corr, k_dim=n_dim)
    return gc

def simulate_from_copula(gc, n_sim):
    return gc.rvs(n_sim)


def invert_uniforms_to_residuals(u_sim, ev_params, empirical_cdfs): #inversing CDF
    n_sim, n_assets = u_sim.shape
    eps_sim = np.zeros_like(u_sim)

    for j, asset in enumerate(ev_params.keys()):
        params = ev_params[asset]
        p_l = params['n_l'] / params['n']
        p_r = params['n_r'] / params['n']
        uL, uR = params['uL'], params['uR']
        sorted_z, ecdf_vals = empirical_cdfs[asset]

        for i in range(n_sim):
            u = u_sim[i, j]
            if u < p_l:
                eps_sim[i, j] = uL - genpareto.ppf(u / p_l, params['c_l'], loc=params['loc_l'], scale=params['scale_l'])
            elif u > 1 - p_r:
                eps_sim[i, j] = uR + genpareto.ppf((1 - u) / p_r, params['c_r'], loc=params['loc_r'], scale=params['scale_r'])
            else:
                adj_u = (u - p_l) / (1 - p_l - p_r)
                eps_sim[i, j] = np.interp(adj_u, ecdf_vals, sorted_z)

    return eps_sim


def simulate_returns(mu, sigma_last, eps_sim):
    ret_sim = np.column_stack([mu[a] + sigma_last[a] * eps_sim[:, j] for j, a in enumerate(mu.keys())]) / 100
    return pd.DataFrame(ret_sim, columns=mu.keys())


def plot_scatter_matrix(df):
    assets = df.columns.tolist()
    n = len(assets)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(df[assets[i]], bins=30)
            else:
                ax.scatter(df[assets[j]], df[assets[i]], s=1)
            if i == n - 1:
                ax.set_xlabel(assets[j])
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(assets[i])
            else:
                ax.set_yticks([])

    plt.suptitle('Matrix of Simulated Returns using Gaussian Copula')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def optimize_cvar(ret_sim, beta=0.95, n_grid=20, solver='CLARABEL'):
    sample_mean = ret_sim.mean(axis=0).values
    q, n = ret_sim.shape
    rho_grid = np.linspace(sample_mean.min(), sample_mean.max(), n_grid)
    cvar_vals = []
    weights = []

    for rho in rho_grid:
        w = cp.Variable(n)
        alpha = cp.Variable()
        u = cp.Variable(q)
        port_loss = -(ret_sim.values @ w)
        constraints = [cp.sum(w) == 1,
                       w >= 0,
                       sample_mean @ w >= rho,
                       u >= 0,
                       u >= port_loss - alpha]
        problem = cp.Problem(cp.Minimize(alpha + (1/(q*(1-beta))) * cp.sum(u)), constraints)
        problem.solve(solver=solver, verbose=False)

        cvar_vals.append(problem.value)
        weights.append(w.value)

    weights_df = pd.DataFrame(weights, index=np.round(rho_grid, 4), columns=ret_sim.columns)
    weights_df.index.name = 'Target Return p'
    return rho_grid, cvar_vals, weights_df #rho_grid, cvar_values, weights_df (index name='Target Return


def optimize_mean_variance(ret_sim, n_grid=20, solver='CLARABEL'):
    mean_ret = ret_sim.mean(axis=0).values
    cov_mat = np.cov(ret_sim.values, rowvar=False)
    n = ret_sim.shape[1]
    rho_grid = np.linspace(mean_ret.min(), mean_ret.max(), n_grid)
    risks, weights = [], []

    for rho in rho_grid:
        w = cp.Variable(n)
        portfolio_var = cp.quad_form(w, cov_mat)
        constraints = [cp.sum(w) == 1,
                       w >= 0,
                       mean_ret @ w >= rho]
        problem = cp.Problem(cp.Minimize(portfolio_var), constraints)
        problem.solve(solver=solver, verbose=False)

        risks.append(np.sqrt(problem.value))
        weights.append(w.value)

    weights_df = pd.DataFrame(weights,
                              index=np.round(rho_grid, 4),
                              columns=ret_sim.columns)
    weights_df.index.name = 'Target Return p'
    return rho_grid, risks, weights_df


if __name__ == '__main__':
    
    # Global Parameters
    symbols = ['JD', 'WNS', 'CRM', 'JPM', 'GPN']
    start_date = '2021-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    n_sim = 1000

    returns = load_returns(symbols, start_date, end_date)
    garch_results, eps_std, sigma_last = fit_gjr_garch(returns)# 2. Fit GJR-GARCH
    ev_params, empirical_cdfs = fit_evt(eps_std)# 3. Fit EVT tails
    uniforms = transform_to_uniforms(eps_std, ev_params, empirical_cdfs)
    gc = fit_gaussian_copula(uniforms)# 5. Fit Gaussian Copula
    u_sim = simulate_from_copula(gc, n_sim)
    eps_sim = invert_uniforms_to_residuals(u_sim, ev_params, empirical_cdfs)
    mu = {a: garch_results[a].params['mu'] for a in symbols}# 8. Simulate from GARCH Copula returns
    df_sim = simulate_returns(mu, sigma_last, eps_sim)
    plot_scatter_matrix(df_sim)

    np.set_printoptions(threshold=np.inf)  #Show all
    
    #Mean-CVaR optimization
    rho_cvar, cvar_vals, weights_cvar = optimize_cvar(df_sim)
    print("Mean-CVaR Weights with target return P:\n", weights_cvar)
    plt.figure()
    plt.plot(rho_cvar, cvar_vals, marker='o')
    plt.xlabel('Expected Return')
    plt.ylabel('CVaR levels')
    plt.title('CVaR Efficient Frontier—Gaussian Copula')
    plt.grid(True)
    plt.show()

    #Mean-Variance optimization
    rho_mv, mv_risks, weights_mv = optimize_mean_variance(df_sim)
    print("standard Mean-Variance Weights with target return P:\n", weights_mv)
    plt.figure()
    plt.plot(mv_risks, rho_mv, 'o-', lw=2)
    plt.xlabel('Portfolio Risk')
    plt.ylabel('Portfolio Return')
    plt.title('FOR COMPARISON - Standard Mean–Variance Efficient Frontier')
    plt.grid(True)
    plt.show()
    
    