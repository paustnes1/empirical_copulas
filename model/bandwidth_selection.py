import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
import time
from datetime import timedelta


from scipy.optimize import minimize
from scipy.special import beta as beta_function
import numba
import scipy.special as sc
import numba_scipy  # The import generates Numba overloads for special
from joblib import Parallel, delayed
from scipy import linalg as LA
import pyvinecopulib as pv
import CRPS.CRPS as pscore
import model.copula_package as copula

numba_scipy._init_extension()
@numba.njit
def cond_empcopulapdf(U=np.array([[]]), h=np.array([]), L=np.array([]), U_known=np.array([])):
    M = U.shape[0]
    D = U.shape[1]
    gridpoints = np.linspace(0, 1, L).repeat(D).reshape((-1, D))                                                                                                                                  
    for i in range(0,D-1):
        gridpoints[:,i+1] = U_known[i]
    cond_pdf=np.zeros(L)
    for ii in range(L):
        kernel_vec = np.zeros(M)
        for jj in range(D):
            kernel_vec+=((gridpoints[ii, jj] / h[jj] + 1)-1)*np.log(U[:,jj]) + (((1 - gridpoints[ii, jj]) / h[jj] + 1)-1)*np.log(1-U[:,jj]) - np.log(sc.beta(gridpoints[ii, jj]/h[jj] + 1, (1-gridpoints[ii, jj])/h[jj] + 1))
        kernel_vec = np.exp(kernel_vec)
        cond_pdf[ii] = np.sum(kernel_vec)/M
    return cond_pdf/np.trapz(cond_pdf,dx=1/L)

@numba.njit
def optimize_loocv_parallel(r, h, L, idx):
    M = r.shape[0]
    D = r.shape[1]
    mask = np.zeros(M, dtype=np.int64) == 0
    mask[idx] = False
    loo = r[mask]
    kernel_vec = np.ones(M-1)
    for jj in range(D):
        kernel_vec *= loo[:,jj]**((r[idx, jj] / h[jj] + 1)-1) * (1-loo[:,jj])**(((1 - r[idx, jj]) / h[jj] + 1)-1) / sc.beta(r[idx, jj]/h[jj] + 1, (1-r[idx, jj])/h[jj] + 1)
    loo_sum = np.sum(kernel_vec)/(M-1)
    return loo_sum

@numba.njit
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    idx = ((array - value)**2).argmin()
    return idx, array[idx]

@numba.njit(parallel=True)
def calc_loo(U, h, L):
    loo_val = 0
    for idx in numba.prange(U.shape[0]):
        loo_val+=optimize_loocv_parallel(U, h, L, idx)
    return loo_val

def ise(h, U, L, U_known):
    """Minimizing the ISE"""
    
    # calculate f_hat squared
    c = cond_empcopulapdf(U, h, L, U_known)
    fhat_squared = np.trapz(np.multiply(c,c),dx=1/L)
    
    # calculate loo
    loo = calc_loo(U, h, L)*(2/(U.shape[0]))
    ise_calc = fhat_squared - loo
    return ise_calc

@numba.njit
def find_inverse_pseudo_known(pseudo_true, true, pseudo_known):
    dim = len(pseudo_known)
    known = []
    for i in range(dim):   
        loc, val_loc = find_nearest(pseudo_true,pseudo_known[i])
        known.append(true[loc])
    return known

quantiles = np.linspace(0.01,0.99,99)
def pl(h, U, L, U_known, rel_params):
    #print(h)
    c = cond_empcopulapdf(U, h, L, U_known)
    forecast = np.zeros(99)
    for i, q in enumerate(quantiles):
            temp1, temp2 = find_nearest(np.cumsum(c)/c.shape[0], q)
            forecast[i] = temp1/c.shape[0]
    #plt.plot(c)
    ploss = 0
    for q in range(99):
        ploss += (rel_params['target_pseudo_known']-forecast[q]) * (quantiles[q]-((rel_params['target_pseudo_known'] <= forecast[q])+0))
    #print(ploss)
    return ploss

def pl_multi(h, U, X, L, U_known, target, h_bool, rel_params, params):
    #print(h)
    ploss = 0
    forecast = np.zeros(99)
    for i in range(params.forecast_horizon):
        c = cond_empcopulapdf(U[f'ts{i}'], h[h_bool[i]], L, np.array(U_known[i]))
        for j, q in enumerate(quantiles):
                temp1, temp2 = find_nearest(np.cumsum(c)/c.shape[0], q)
                forecast[j] = temp1/c.shape[0]
        #plt.plot(c)
        all_locs = []
        for jj in np.linspace(0.01,0.99,99):
            temp1, temp2 = find_nearest(np.cumsum(c)/c.shape[0], jj)
            all_locs.append(temp1/c.shape[0])
        all_values = find_inverse_pseudo_known(U[f'ts{i}'][:,0], X[f'ts{i}'][:,0], all_locs)
        for q in range(99):
            ploss += (target[i]-all_values[q]) * (quantiles[q]-((target[i] <= all_values[q])+0))
        #print(ploss)
    #ploss /= 99*params.forecast_horizon
    print(f'ploss: {ploss}. h: {h}')
    return ploss

def ise_fixed(h, U, L, U_known, h_fixed):
    """Minimizing the ISE"""
    h_comb = np.insert(h_fixed, -1, h)
    # calculate f_hat squared
    c = cond_empcopulapdf(U, h_comb, L, U_known)
    fhat_squared = np.trapz(np.multiply(c,c),dx=1/L)
    
    # calculate loo
    loo = calc_loo(U, h_comb, L)*(2/(U.shape[0]))
    ise_calc = fhat_squared - loo
    return ise_calc

def optimize_bandwidth(U, U_known, h0, rel_params, params):
    L = params.L

    bounds = Bounds(1e-3*np.ones(U.shape[1]), 1*np.ones(U.shape[1]))
    t0 = time.time()    

    if(params.optimization_objective=='ise'):
        res = minimize(ise, h0, method='L-BFGS-B',
                args=(U, L, U_known),
                options={'disp': True, 'ftol': 1e-6,'eps':1e-6}, bounds=bounds)
        
    elif(params.optimization_objective=='ise_fixed'):
        bounds = Bounds(1e-3*np.ones(1), 1*np.ones(1))
        h0 = 0.1
        sel_date = str(rel_params['sel_date'].date())
        fixed_bws = pd.read_csv(f'optimized_bandwidths/ise/{sel_date}_optimal_bandwidths.csv', index_col=0).values
        res = minimize(ise_fixed, h0, method='L-BFGS-B',
                args=(U, L, U_known, fixed_bws[rel_params['ts']]),
                options={'disp': True, 'ftol': 1e-6,'eps':1e-6}, bounds=bounds)

    elif(params.optimization_objective=='pl'):
        print(f'h0: {h0}')
        res = minimize(pl, h0, method='L-BFGS-B',
                args=(U, L, U_known, rel_params),
                options={'disp': True, 'ftol': 1e-6}, bounds=bounds)
        print(res)
        print(res.x)
        #plt.show()

    elif(params.optimization_objective=='pl_multi'):
        print('Bandwidth selection objective: pl_multi')
        # Since we go 7 days back in time, the conitioning dataset actually also contains the true outcomes. Fine, since we are BW optimizing.
        # However, we must adjust the training set...
        forecast_horizon = params.forecast_horizon
        sel_date = rel_params['sel_date']-timedelta(days=7)
        rel_params['sel_date'] = sel_date
        print(sel_date)
        rel_params['training'] = rel_params['training'][rel_params['training'].index.date<sel_date.date()].copy()

        u_multi = {}
        x_multi = {}
        pseudo_known_multi = []
        target_multi = []
        h_bool = []
        for i in range(params.forecast_horizon):
            rel_params['relative_ts'] = sel_date+timedelta(minutes=i*rel_params['sample_period_minutes'])
            rel_params['ts'] = i
            relative_lags, lags_to_keep = copula.get_relative_lags(rel_params['training'], i, rel_params, params)
            boolean_array = [element in lags_to_keep for element in params.fixed_lags]
            #adding the forecast dimension
            boolean_array.insert(0, True)
            #adding temperature dimension
            boolean_array.insert(len(boolean_array), True)
            h_bool.append(boolean_array)
            #print(relative_lags)
            rel_params['relative_lags'] = relative_lags
            target_multi.append(rel_params['conditioning']['p'].loc[rel_params['relative_ts']])
            u, x, pseudo_known = copula.build_copula(rel_params['training'], rel_params['conditioning'], rel_params, params)
            #print(u.shape)
            u_multi[f'ts{i}'] = u
            x_multi[f'ts{i}'] = x
            pseudo_known_multi.append(pseudo_known)
        target_multi = np.array(target_multi)
        print(h_bool[0])
        # We define the maximum amount of dimensions (number of lags + forecast dimension + temperature dimension)
        max_dim = len(params.fixed_lags)+2
        print(max_dim)
        if(h0 is None):
            h0 = 0.01*np.ones(max_dim)
            print(h0)
        bounds = Bounds(1e-3*np.ones(max_dim), 1*np.ones(max_dim))
        res = minimize(pl_multi, h0, method='L-BFGS-B',
            args=(u_multi, x_multi, L, pseudo_known_multi, target_multi, h_bool, rel_params, params),
            options={'disp': True, 'eps': 1e-3}, bounds=bounds)
        print(res)

    else:
        print('Error. Optimization-objective not available')
    t1 = time.time()
    total = t1-t0
    print(f'total time: {total}')
    if 0:
        e_vals, e_vecs = LA.eig(res.hess_inv.todense())
        print('Eigenvalues of hessian')
        print(e_vals)
    return res
