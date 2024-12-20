'''
Probabilistic forecasts based on empirical copulas.
copula_paclage version 2.0
Date: 20.12.2024
------------------
- 


Author: Pål Forr Austnes
'''

from math import exp, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from datetime import date
from datetime import timedelta
import os
import pyvinecopulib as pv
from workalendar.registry import registry
from tqdm import tqdm
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
import importlib

'''
Data loading: replace with your own data loading function
Format for the code must be a dataframe with at least a column named 'p' for power
'''
def load_data(id_data):
    nwp_data = pd.read_hdf("Rolle_dataset/nwp_data.h5","df")
    power_data = pd.read_pickle("Rolle_dataset/power_data.p")
    idx = power_data['P_mean'].columns
    df = power_data['P_mean'][[idx[id_data]]].copy()
    # Take first forecast for every day and use it as the temperature forecast for the day
    new_ind = []
    new_data = []
    for ind in nwp_data.index:
        if(ind.hour==0):
            if(ind.minute==10):
                new_ind.append(ind)
                new_data.append(nwp_data.loc[ind]['temperature'])
    arr = np.array(new_data).flatten()
    t = pd.DataFrame(data=arr,index=pd.date_range(start=new_ind[0],end='2019-01-19 23:50:00+00:00', freq='1H'), columns=['t'])
    date_index2 = pd.date_range(start=new_ind[0],end='2019-01-20 00:10:00+00:00', freq='1H')
    t = t.reindex(date_index2)
    t = t.resample('10min').ffill()
    t = t[0:-2]
    # Add temperature
    df['t'] = t['t']
    df.columns=['p','t_meas']
    df['t_forecast'] = df['t_meas']
    df = df['2018-01-14':'2019-01-19']
    #plt.scatter(df['p'],df['t_meas'])
    dataset = df.copy()
    dataset = dataset.resample('15min').mean()
    return dataset, idx[id_data]

'''
Misc. functions: TODO: Move to a separate file
'''
def find_pseudo_known(x, known):
    dim = x.shape[1]
    pseudo_known = []
    for i in range(1,dim):
        temp = np.append(x[:,i],known[i-1])
        pseudo_known.append(pv.to_pseudo_obs(temp)[-1])
    return pseudo_known


def cond_empcopulapdf(U, h, U_known, rel_params, params):
    M = U.shape[0]
    D = U.shape[1]
    L = params.L
    cond_pdf = np.zeros(L)
    gridpoints = np.tile(np.linspace(0, 1, L), (D, 1)).T
    gridpoints[:, range(1, D)] = U_known

    kernel_vecs = np.ones((L, M))
    for ii in range(L):
        kernel_vecs[ii, :] = np.exp(
            np.sum(
                beta.logpdf(
                    U, gridpoints[ii, :] / h + 1, (1 - gridpoints[ii, :]) / h + 1
                ),
                axis=1,
            )
        )
    cond_pdf = np.sum(kernel_vecs, axis=-1) / M
    return cond_pdf

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_inverse_pseudo_known(pseudo_true, true, pseudo_known):
    dim = len(pseudo_known)
    known = []
    for i in range(dim):   
        loc, val_loc = find_nearest(pseudo_true,pseudo_known[i])
        known.append(true[loc])
    return known

def create_constant_copula(training, ts, rel_params, params):
    hour = rel_params['relative_ts'].hour
    minute = rel_params['relative_ts'].minute
    x = pd.DataFrame()
    x['target'] = training['p'][(training.index.hour==hour) & (training.index.minute==minute)]

    for i, lag in enumerate(rel_params['relative_lags']):
        hour = lag.hour
        minute = lag.minute
        delta_day = (rel_params['sel_date'].date()-lag.date()).days
        x[f'lag{i}-{hour}:{minute}_Dayshift:{delta_day}'] = training['p'][(training.index.hour==hour) & (training.index.minute==minute)].shift(delta_day).values
    return x

def add_temperature(x,dataset, ts, rel_params, params):
    timestamp = rel_params['sel_date']+timedelta(hours=(ts))
    val = dataset['t_meas'][(dataset.index.hour==timestamp.hour) & (dataset.index.minute==timestamp.minute)].values
    x[f't{ts}'] = val
    return x

def add_weekday(x, dataset, timestep):
    return np.c_[x, dataset['weekday'][(dataset.index.hour==dayindex[timestep,0]) & (dataset.index.minute==dayindex[timestep,1])].values]

def add_ghi(x, dataset, ts, sel_date):
    timestamp = sel_date+timedelta(hours=(ts))
    val = dataset['GHI_measured'][(dataset.index.hour==timestamp.hour) & (dataset.index.minute==timestamp.minute)].values
    x[f'ghi{ts}'] = val
    return x

def show(results, rel_params, params):
    try:
        copula_forecast = results['forecast']
    except:
        copula_forecast = None
    try:
        copula_target = results['target']
    except:
        copula_target = None
    try:
        copula_expected_value = results['expected_value']
    except:
        copula_expected_value = np.nan*np.ones(len(copula_target))
    ploss = [] 
    quantiles = np.linspace(0.01,0.99,99)
    for q in range(99):
        ploss.append( (copula_target[:]-copula_forecast[:,q]) * (quantiles[q]-((copula_target[:] <= copula_forecast[:,q])+0))   )
    ploss = np.mean(ploss)

    try:
        exp_rmse = sqrt(mean_squared_error(copula_target, copula_expected_value))
    except:
        exp_rmse = np.nan
        print('Failed calculating the expected value. Something might be wrong')
    time = pd.date_range(start=rel_params['sel_date'], periods=params.forecast_horizon, freq=rel_params['sample_period'])
    if(len(time)!=len(copula_target)):
        time = np.arange(len(copula_target))
    fig, ax=plt.subplots(figsize=(10,6))
    ax.plot(time, copula_target, c='green', label='Power measured at node')
    ax.plot(time, copula_forecast[:,9],'--', c='red', label=f'Forecast: 10-90% | QL: {np.round(ploss,2)}')
    ax.plot(time, copula_forecast[:,89],'--', c='red')
    ax.plot(time, copula_expected_value, c='pink', label=f'Expected value (RMSE: {np.round(exp_rmse,2)})')
    color = np.linspace(1,0.1,49)
    for i in range(1,49):
        ax.fill_between(time,copula_forecast[:,50-i], copula_forecast[:,50+i], facecolor='darkblue',alpha=color[i])

    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlabel('HH:MM')
    forecast_date = rel_params['sel_date']
    plt.xlabel(f'{str(forecast_date.date())}  ({forecast_date.strftime("%A")})', fontsize=14)
    plt.ylabel(f'Power [{params.power_unit}]', fontsize=14)
    plt.title(f'{params.plot_string}', fontsize=14)
    ax.legend(loc='upper left')
    try:
        ax.set_ylim([params.fig_ymin, params.fig_ymax])
    except:
        _=0
    plot_location = 'generated_plots/'
    #plt.savefig(f'{plot_location}{plot_string}_forecast_{str(date.date())}.png', bbox_inches='tight',dpi=(250))
    return fig

def get_relative_lags(df, ts, rel_params, params):
    CalendarClass = registry.get('CH-VD')
    calendar = CalendarClass()
    pred_step = rel_params['sel_date'] + timedelta(minutes=ts*rel_params['sample_period_minutes'])
    relative_lags = []
    lags_to_keep = []

    for j, i in enumerate(params.fixed_lags):
        sel_date_temp = rel_params['sel_date']-timedelta(minutes=(i-ts)*rel_params['sample_period_minutes'])
        if(i<10):
            relative_lags.append(sel_date_temp)
            lags_to_keep.append(i)
            continue
        if(calendar.is_working_day(pred_step) and not calendar.is_working_day(sel_date_temp)):
            continue
        if(not calendar.is_working_day(pred_step) and calendar.is_working_day(sel_date_temp)):
            continue
        else:    
            relative_lags.append(sel_date_temp)
            lags_to_keep.append(i)
    return relative_lags, lags_to_keep

def build_copula(training, conditioning, rel_params, params):
    ts = rel_params['ts']
    x = create_constant_copula(training, ts, rel_params, params)

    # EXOGENEOUS VARIABLES
    if(params.include_temperature):
        x = add_temperature(x, training, ts, rel_params, params)
    if(params.include_weekday):
        x = add_weekday(x, training, ts, rel_params, params)
    if(params.include_ghi):
        x = add_ghi(x,training,ts, rel_params, params)
    x = precluster_data(x, rel_params, params)
    x = x.dropna()
    x = x.values

    if(len(rel_params['relative_lags'])==0):
        cond_fixed = np.empty(0)
    else:
        cond_fixed = []
        for i in rel_params['relative_lags']:
            cond_fixed.append(conditioning['p'].loc[str(i)])
        cond_fixed = np.array(cond_fixed)

    # EXOGENEOUS VARIABLES
    if(params.include_temperature):
        cond_fixed = np.append(cond_fixed, conditioning['t_meas'].loc[rel_params['relative_ts']])
    if(params.include_weekday):
        cond_fixed = np.append(cond_fixed, known_date[ts])
    if(params.include_ghi):
        cond_fixed = np.append(cond_fixed, known_ghi[ts])
    
    pseudo_known = find_pseudo_known(x, cond_fixed)
    pseudo_known = np.array(pseudo_known).flatten()

    u = pv.to_pseudo_obs(x)
    return u, x, pseudo_known

def calc_dayahead(training, conditioning, rel_params, params):
    save_optimized_bandwidths = []
    expected_value = []
    save_big = []
    result = []
    forecast = []
    cols = 0
    rows = 0
    h_opts = []    
    # Load precalculated bandwidths
    sel_date = rel_params['sel_date']
    if(params.h is not None):
        print('Using user-specified bandwidths')
    else:
        df_columns = ['predict_hour']
        df_columns += [str(i) for i in params.fixed_lags]
        if(params.include_temperature):
            df_columns.append('temperature')
        if(params.include_ghi):
            df_columns.append('ghi')
        df_columns = '_'.join(df_columns)
        try:
            bw_saved = pd.read_csv(f'optimized_bandwidths/ise/{params.data_id}_{str(sel_date.date())}_optimal_bandwidths_{str(df_columns)}.csv', index_col=0).values
            print(f'Precalculated bandwidths available')
        except:
            print(f'Precalculated bandwidths not available')
            bw_saved = None
    # ----------------- MAIN LOOP -------------------#
    for ts in range(0,params.forecast_horizon):
        relative_ts = rel_params['sel_date']+timedelta(minutes=ts*rel_params['sample_period_minutes'])
        rel_params['relative_ts'] = relative_ts
        rel_params['ts'] = ts
        # Create a list of lags. This one picks all lags up to nb_lags
        relative_lags, lags_to_keep = get_relative_lags(training, ts, rel_params, params)
        boolean_array = [element in lags_to_keep for element in params.fixed_lags]

        #adding the forecast dimension
        boolean_array.insert(0, True)

        #adding temperature dimension
        '''
        Todo: split this into a separate function
        '''
        boolean_array.insert(len(boolean_array), True)
        rel_params['relative_lags'] = relative_lags
        u, x, pseudo_known = build_copula(training, conditioning, rel_params, params)
        if(params.h is not None):
            h_to_keep = params.h[boolean_array]
        elif(params.find_bandwidth):
            import model.bandwidth_selection as bw_selection
            importlib.reload(bw_selection)

            if(params.x0 is None):
                x0 = 0.01*np.ones((params.forecast_horizon, u.shape[1]))
            else:
                x0 = params.x0

            x_space = np.append(x[:,0], rel_params['targets'].loc[rel_params['relative_ts']])
            rel_params['target_pseudo_known'] = pv.to_pseudo_obs(x_space)[-1]
            sel_date = rel_params['sel_date']

            # ---------- If the objective is over several time steps, we need to add all the data -------------#
            if(params.optimization_objective=='pl_multi'):
                rel_params['training'] = training
                rel_params['conditioning'] = conditioning
            if(params.optimization_objective=='crps_multi'):
                rel_params['training'] = training
                rel_params['conditioning'] = conditioning

            res = bw_selection.optimize_bandwidth(u, pseudo_known, x0[ts], rel_params, params)
            if(params.optimization_objective=='ise_fixed'):
                sel_date = str(rel_params['sel_date'].date())
                fixed_bws = pd.read_csv(f'optimized_bandwidths/ise/{sel_date}_optimal_bandwidths.csv', index_col=0).values
                h_to_keep = np.insert(fixed_bws[rel_params['ts']], -1, res.x)
                save_optimized_bandwidths.append(h_to_keep)
            else:
                h_to_keep = res.x
                save_optimized_bandwidths.append(res.x)
        elif(bw_saved is not None):
            h_to_keep = bw_saved[ts]
            h_to_keep = h_to_keep[~np.isnan(h_to_keep)]
        else:
            h_to_keep = np.array(0.1*np.ones(u.shape[1]))
        # present conclusion of variables:
        if 0:
            print(f'Number of dimensions of copula: {u.shape}')
            print(f'... corresponding to the following relative lags: {relative_lags}')
            print(f'Bandwidths are: {h_to_keep}')

        if(params.rule_of_thumb_bandwidth):
            h_to_keep = rule_of_thumb_bandwidth(u)
        c_cond_quick = cond_empcopulapdf(u, h_to_keep, pseudo_known, rel_params, params)
        cop_size = len(c_cond_quick)
        area = np.trapz(c_cond_quick,np.linspace(0,1,cop_size))
        c_cond_quick /= area

        if(params.output == 'scenarios'):
            rel_params[f'x{ts}'] = x[:,0]
            rel_params[f'u{ts}'] = u[:,0]
            rel_params[f'pdf{ts}'] = c_cond_quick

        # Present and store result
        e_cond = np.sum(np.multiply(np.linspace(0,1,cop_size),c_cond_quick/cop_size))
        e_cond = find_inverse_pseudo_known(u[:,0], x[:,0], [e_cond,e_cond])
        c05, val05 = find_nearest(np.cumsum(c_cond_quick)/cop_size, 0.05)
        c95, val95 = find_nearest(np.cumsum(c_cond_quick)/cop_size, 0.95)
        c05 /= cop_size
        c95 /= cop_size

        # Resampling:
        sample = np.random.choice(np.linspace(0, 1, params.L), size=1, replace=True, p=c_cond_quick/sum(c_cond_quick))
        true_locations = find_inverse_pseudo_known(u[:,0], x[:,0], [sample])
        result.append(true_locations[0])

        conditioning.loc[rel_params['sel_date']+timedelta(minutes=ts*rel_params['sample_period_minutes']), 'p'] = true_locations[0]

        all_locs = []
        for jj in np.linspace(0.01,0.99,99):
            temp1, temp2 = find_nearest(np.cumsum(c_cond_quick)/cop_size, jj)
            all_locs.append(temp1/cop_size)
        all_values = find_inverse_pseudo_known(u[:,0], x[:,0], all_locs)
        forecast.append(all_values)            
        save_big.append(c_cond_quick)
        #save_big.append(all_values)
        expected_value.append(e_cond[0])
        
    results = {'forecast': np.array(forecast),
               'expected_value': np.array(expected_value),
               'optimal_bandwidths':np.array(h_opts),
               'save_optimized_bandwidths':save_optimized_bandwidths,
               'forecast_pdf' : np.array(save_big)
               }
    return results, rel_params

## Evaluation
# Todo: split into a separate file
def pinball_loss(y_true, y_pred, tau):
    """
    Calculates the Pinball loss for a given quantile tau, between the true and predicted values.
    
    Parameters
    ----------
    y_true : array-like
        The true values.
    y_pred : array-like
        The predicted values.
    tau : float
        The quantile to calculate the loss for, should be between 0 and 1.
    
    Returns
    -------
    float
        The Pinball loss for the given quantile tau.
    """
    delta = y_true - y_pred
    return np.where(delta >= 0, tau * delta, (tau - 1) * delta).mean()

def calculate_pinball_loss(targets,predictions):
    predictions = np.array(predictions)
    targets = np.array(targets)
    quantiles = np.linspace(0.01,0.99,predictions.shape[1])
    val = []
    for i in range(targets.shape[0]):
        for j in range(len(quantiles)):
            
            val.append(pinball_loss(targets[i], predictions[i,j], quantiles[j]))
    return np.mean(val)

def metrics(test, ci_lower, ci_higher, forecast_quantiles, verbose=False):
    import properscoring as ps
    from scipy.interpolate import interp1d
    # Calculate PICP:
    picp=0
    for i in range(len(test)):
        if(test[i]<ci_higher[i] and test[i]>ci_lower[i]):
            picp=picp+1
    picp = picp/len(test)

    # Calculate PINAW:
    N = len(ci_higher)
    confidence = np.subtract(ci_higher,ci_lower)
    #K = max(confidence)
    K = max(test)-min(test)
    pinaw = 1/(N*K)*np.sum(confidence)

    # Calculate CWC:
    alpha=0.8
    if(picp<=alpha):
        eta=1
    else:
        eta=0
    nu = 50
    cwc = pinaw*(1+eta*picp*np.exp(-nu*(picp-alpha)))

    # Calculate CRPS:
    crps=0
    quantiles = np.linspace(0.01,0.99,99)
    for i in range(forecast_quantiles.shape[0]):
        idx, val = find_nearest(forecast_quantiles[i,:], test[i])
        crps += np.sum(np.square(quantiles[0:idx+1]))+np.sum(np.square(quantiles[idx+1:]-1))
    crps /= forecast_quantiles.shape[0]

    # CRPS Properscore quadrature:
    crps_quadrature = 0
    '''
    for i in range(forecast_quantiles.shape[0]):
        cdf_function = interp1d(forecast_quantiles[i,:], np.linspace(0.01,0.99,99), bounds_error=False, fill_value=(-np.inf, np.inf))

        crps_quadrature += ps.crps_quadrature(test[i], cdf_function, xmin=None, xmax=None, tol=2)
    crps_quadrature /= forecast_quantiles.shape[0]
    '''

    if(verbose):
        print(f'PICP: {np.round(picp, 3)}')
        print(f'PINAW: {np.round(pinaw, 3)}')
        print(f'CWC: {np.round(cwc, 3)}')
        print(f'CRPS: {np.round(crps,3)}')
        print(f'CRPS (properscore ensemble): {np.round(np.mean(ps.crps_ensemble(test, forecast_quantiles)),2)}')
        print(f'CRPS (properscore quandrature): {np.round(crps_quadrature,3)}')
    return picp,3, pinaw, cwc, crps

def check_input_data(data, params):
    hour = np.array([[j for i in range(4)] for j in range(24)]).flatten()
    minute = np.array([[i for i in range(0,60,15)] for j in range(24)]).flatten()
    dayindex = np.array((hour,minute)).T
    try:
        temp = data['p']
    except:
        print('Input error, please provide power data in a column named "p"')

    if(params.include_weekday):
        try:
            temp = data['weekday']
        except:
            print('Input error, please provide weekday data in a column named "weekday"')
    if(params.include_temperature):
        try:
            temp = data['t_meas']
            temp = data['t_forecast']
        except:
            print('Input error, please provide temperature data in columns named "t_meas and t_forecast"')
    counts=[]
    for i in range(24):
        counts.append(len(data['p'][data.index.hour==i]))
    if(not counts.count(counts[0]) == len(counts)):
        print('Input error. Data doesnt have equal number of samples. Verify that timeseries is complete, contains full days of data and has correct frequency.')
        print('Trying to reindex time series...')
        new_index = pd.date_range(start=data.index[0],end=data.index[-1], freq=params.sample_period)
        data = data.reindex(new_index)
    
    #print('Input data OK')
    return data

def precluster_data(data, rel_params, params):
    CalendarClass = registry.get('CH-VD')
    calendar = CalendarClass()
    mask1 = [calendar.is_working_day(i) for i in pd.to_datetime(data.index)]

    # Set saturdays to working day
    if(params.saturday_is_weekday):
        for i, index in enumerate(data.index):
            if(index.dayofweek==5):
                mask1[i] = True
    
    mask2 = [not i for i in mask1]
    if(params.saturday_is_weekday):
        if(calendar.is_working_day(rel_params['relative_ts']) or rel_params['relative_ts'].weekday()==5):
            data = data[mask1].copy()
            day_before = rel_params['relative_ts'] +timedelta(-1)
            while(not calendar.is_working_day(day_before) or day_before.weekday()==5):
                day_before = day_before +timedelta(-1)
        else:
            data = data[mask2].copy()
            day_before = rel_params['relative_ts'] +timedelta(-1)
            while(day_before.weekday()!=6):
                day_before = day_before +timedelta(-1)
    else:
        if(calendar.is_working_day(rel_params['relative_ts'])):
            data = data[mask1].copy()
            day_before = rel_params['relative_ts'] +timedelta(-1)
            while(not calendar.is_working_day(day_before)):
                day_before = day_before +timedelta(-1)
        else:
            data = data[mask2].copy()
            day_before = rel_params['relative_ts'] +timedelta(-1)
            while(calendar.is_working_day(day_before)):
                day_before = day_before +timedelta(-1)

    return data

def normalize_data(data_conditioning, data_training):
    mean_vector = data_training.mean(axis=0)
    data_training = data_training-mean_vector
    data_conditioning = data_conditioning- mean_vector
    return data_conditioning, data_training, mean_vector


'''
nan_handler ffills nans if the number of consequtive nans are less than max_nan AND removes days with more than max_nan number of nans
'''
def nan_handler(dataset, max_nan):
    if(dataset['p'].isna().sum()>0):
        nan_timestamps = dataset.index[np.isnan(dataset['p'].values)]
        nandays = [i.date() for i in nan_timestamps]
        val, counts = np.unique(nandays, return_counts=True)
        val = val[counts>max_nan]

        dataset = dataset.ffill()

        val = [str(i) for i in val]
        for i in val:
            dataset.loc[i] = np.nan
            
        dataset = dataset.dropna()
        print(f'Nan-handler removed {len(val)} days with too much missing data. The rest of nan-values was ffilled')
    return dataset

'''
nan_handler2 ffills nans if the number of consequtive nans are less than max_nan
'''
def nan_handler2(dataset, max_nan):
    if(dataset['p'].isna().sum()>0):
        nan_timestamps = dataset.index[np.isnan(dataset['p'].values)]
        nandays = [i.date() for i in nan_timestamps]
        val, counts = np.unique(nandays, return_counts=True)
        val = val[counts<max_nan]

        val = [str(i) for i in val]
        for i in val:
            dataset.loc[i] = dataset.loc[i].ffill()
    return dataset

def detrend_data(conditioning, training, sel_date, forecast_horizon):
    from scipy import stats
    # linear regression without nans
    y = training['p'].values
    x = np.arange(len(y))
    ind = ~np.isnan(y)
    m, b, r_val, p_val, std_err = stats.linregress(x[ind],y[ind])
    detrend_y = y - (m*x + b)
    if 1:
        p = np.polyfit(x[ind], y[ind], 2)
        detrend_yy = y - (p[0]*x**2 + p[1]*x + p[2])

    training['p'] = detrend_y
    index_start = len(training[training.index.date<sel_date])
    x_trend_test = np.arange(index_start, index_start+forecast_horizon) 
    x_trend_conditioning = np.arange(len(conditioning))
    conditioning['p'] = conditioning['p'] - (m*x_trend_conditioning + b)

    return conditioning, training, m, b, x_trend_test

def run_prediction(df, sel_date, params):
    rel_params = {'sample_period' : df.index.inferred_freq}
    rel_params['sel_date'] = sel_date
    date_range = pd.date_range(start=sel_date, periods = params.forecast_horizon, freq=rel_params['sample_period'], tz='utc')

    if(params.find_bandwidth):
        rel_params['targets'] = df['p'].loc[date_range]
    df = check_input_data(df, params)
    df = nan_handler2(df, 6)

    if(rel_params['sample_period']=='15T'):
        rel_params['sample_period_minutes'] = 15
    elif(rel_params['sample_period']=='H'):
        rel_params['sample_period_minutes'] = 60
    else:
        print('Sample frequency not supported')
        print(df.index.inferred_freq)

    conditioning = df.loc[np.concatenate([df.index[df.index<sel_date], date_range])].copy()
    conditioning.loc[conditioning.index>=sel_date, 'p'] = np.nan

    training = df[df.index.date<sel_date.date()].copy()

    if(params.remove_mean):
        print('Mean from every timestep is removed')
        conditioning, training, mean_vector = normalize_data(conditioning, training)
    # Detrend data
    if(params.detrend_timeseries):
        conditioning, training, slope, intercept, x_trend = detrend_data(conditioning, training, sel_date, params)
        print('Data is detrended')
    else:
        slope = np.nan
        intercept = np.nan
        x_trend = np.nan

    if(params.find_bandwidth):
        if((params.optimization_objective == 'pl_multi') or (params.optimization_objective == 'crps_multi')):
            import model.bandwidth_selection as bw_selection
            importlib.reload(bw_selection)
            rel_params['training'] = training
            rel_params['conditioning'] = conditioning
            # Just dummy to get values
            rel_params['ts'] = 0
            rel_params['relative_ts'] = rel_params['sel_date']
            rel_params['relative_lags'] = [rel_params['sel_date']-timedelta(days=1), rel_params['sel_date']-timedelta(days=7)]
            u, x, pseudo_known = build_copula(rel_params['training'], rel_params['conditioning'], rel_params, params)
            res = bw_selection.optimize_bandwidth(u, pseudo_known, None, rel_params, params)
            save_date = rel_params['sel_date']
            np.save(f'optimized_bandwidths/pl/optimal_bws_{params.fixed_lags}_{str(save_date.date())}', res)
            return None, None
    
    results, rel_params = calc_dayahead(training, conditioning, rel_params, params)
    if(params.find_bandwidth):
        df_columns = ['predict_hour']
        df_columns += [str(i) for i in params.fixed_lags]
        if(params.include_temperature):
            df_columns.append('temperature')
        if(params.include_ghi):
            df_columns.append('ghi')
        df_columns = '_'.join(df_columns)
        save_optimized_bandwidths_pd = pd.DataFrame(results['save_optimized_bandwidths'])
        if(params.optimization_objective == 'pl'):
            path = 'optimized_bandwidths/pl/'
            save_optimized_bandwidths_pd.to_csv(path+f'{str(sel_date.date())}_optimal_bandwidths.csv')
        elif(params.optimization_objective == 'ise'):
            path = 'optimized_bandwidths/ise/'
            #save_optimized_bandwidths_pd.to_csv(path+f'{str(sel_date.date())}_optimal_bandwidths.csv')
            save_optimized_bandwidths_pd.to_csv(path+f'{str(sel_date.date())}_optimal_bandwidths_{str(df_columns)}.csv')
        elif(params.optimization_objective == 'ise_fixed'):
            path = 'optimized_bandwidths/ise_fixed/'
            save_optimized_bandwidths_pd.to_csv(path+f'{str(sel_date.date())}_optimal_bandwidths_{str(df_columns)}.csv')
        elif(params.optimization_objective == 'crps'):
            path = 'optimized_bandwidths/crps/'
            save_optimized_bandwidths_pd.to_csv(path+f'{str(sel_date.date())}_optimal_bandwidths.csv')
        elif(params.optimization_objective == 'pl_multi'):
            path = 'optimized_bandwidths/pl_multi/'
            save_optimized_bandwidths_pd.to_csv(path+f'{str(sel_date.date())}_optimal_bandwidths.csv')
    
    #print(h_opts)
    if(params.remove_mean):
        res = np.array(res)
        res = res + mean_vector['p']
    if(params.detrend_timeseries):
        res = np.array(res)
        for i in range(res.shape[1]):
            res[:,i] = res[:,i] + (slope*x_trend + intercept)
        expected_value = expected_value + (slope*x_trend + intercept)

    results['target'] = df['p'].loc[date_range]
    return results, rel_params

        
def sample_from_scenarios(data, n_samples, forecast_horizon, sel_date, L, nb_days):
    import pathlib
    print(pathlib.Path().resolve())
    import os


    from tqdm import tqdm
    samples = np.zeros((forecast_horizon,n_samples))
    for step in tqdm(range(forecast_horizon)):
        rel_day = sel_date+timedelta(hours=step)
        training1 = pd.read_csv(f'training_{str(rel_day.date())}.csv', index_col=0, parse_dates=True)
        true = create_constant_copula(training1, step, [], sel_date+timedelta(hours=step), [])
        true=true.values
        r_idx = np.random.choice(range(data.shape[0]), size=n_samples)
        for i, idx_scenario in enumerate(r_idx):
            sample = np.random.choice(np.linspace(0,1,L), size=1, replace=True, 
                                    p=data[idx_scenario,step,:]/sum(data[idx_scenario,step,:]))
            
            pseudo_true = pv.to_pseudo_obs(true) # move outside
            
            sample_true = find_inverse_pseudo_known(pseudo_true[:,0], true[:,0], sample)[0]
            samples[step,i] = sample_true

    return samples

def rule_of_thumb_bandwidth(data):
    cov_matrix = np.cov(data,rowvar=False)
    nb_dim = data.shape[1]
    nb_samples = data.shape[0]

    # Computing diagonalization from here : https://stackoverflow.com/questions/61262772/is-there-any-way-to-get-a-sqrt-of-a-matrix-in-numpy-not-element-wise-but-as-a
    #evalues, evectors = np.linalg.eig(cov_matrix)
    ## Ensuring square root matrix exists
    #assert (evalues >= 0).all()
    #sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)

    # More clean version using scipy directly. Uses Schur's decomposition. https://link.springer.com/chapter/10.1007/978-3-642-36803-5_12
    from scipy.linalg import sqrtm
    sqrt_matrix = sqrtm(cov_matrix)

    # rule of thumb
    sqrt_variances = np.array([sqrt_matrix[i,i] for i in range(nb_dim)])
    # only variance of individual dimensions
    sqrt_variances = np.std(data,axis=0)

    h = ((4)/(nb_dim+2))**(1/(nb_dim+4))*(1/nb_samples**(1/(nb_dim+4))) * sqrt_variances
    #print(f'Rule of thumb bandwidth: {h}')
    return h