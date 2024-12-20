'''
QR code for SEGAN paper submission

Author: Pål Forr Austnes

'''
import numpy as np
import pandas as pd
from workalendar.registry import registry
from datetime import datetime,timedelta,date
import statsmodels.regression.quantile_regression as sm
import matplotlib.pyplot as plt


def precluster_data(data, sel_date, saturday_is_weekday):
    CalendarClass = registry.get('CH-VD')
    calendar = CalendarClass()
    mask1 = [calendar.is_working_day(i) for i in pd.to_datetime(data.index)]
    #mask1 = data.index.dayofweek<5
    
    # Set saturdays to working day
    if(saturday_is_weekday):
        for i, index in enumerate(data.index):
            if(index.dayofweek==5):
                mask1[i] = True
    
    mask2 = [not i for i in mask1]

    if(saturday_is_weekday):
        if(calendar.is_working_day(sel_date) or sel_date.weekday()==5):
            data = data[mask1].copy()
            day_before = sel_date +timedelta(-1)
            while(not calendar.is_working_day(day_before) or day_before.weekday()==5):
                day_before = day_before +timedelta(-1)
        else:
            data = data[mask2].copy()
            day_before = sel_date +timedelta(-1)
            while(day_before.weekday()!=6):
                day_before = day_before +timedelta(-1)
    else:
        if(calendar.is_working_day(sel_date)):
            data = data[mask1].copy()
            day_before = sel_date +timedelta(-1)
            while(not calendar.is_working_day(day_before)):
                day_before = day_before +timedelta(-1)
        else:
            data = data[mask2].copy()
            day_before = sel_date +timedelta(-1)
            while(calendar.is_working_day(day_before)):
                day_before = day_before +timedelta(-1)
    #print(f'Data is clustered into weekend and weekday, the day is {sel_date.strftime("%A")}. The day before is: {day_before.strftime("%A")}')
    if(saturday_is_weekday):
        print(f'Data is clustered into working day and not-working day, (Note! Saturday is considered weekday). The day is {sel_date.strftime("%A")}. The day before is: {day_before.strftime("%A")}')
    else:
        print(f'Data is clustered into working day and not-working day, (Note! Saturday is considered weekend). The day is {sel_date.strftime("%A")}. The day before is: {day_before.strftime("%A")}')
    print(f'The data input size is: {np.shape(data)}')
    return data, day_before

def show(arr, date, target, plot_string, forecast_horizon):
    import matplotlib.dates as mdates

    time = pd.date_range(start=date, periods=forecast_horizon,freq='15Min')
    arr = np.array(arr)
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(time, target, c='green', label='Power measured at node')
    ax.plot(time, arr[:,49], c='red', label='Forecast: expected value')
    ax.plot(time, arr[:,9],'--', c='red', label='Forecast: 10-90%')
    ax.plot(time, arr[:,89],'--', c='red')
    color = np.linspace(1,0.1,49)
    for i in range(1,49):
        ax.fill_between(time,arr[:,50-i],arr[:,50+i],facecolor='darkblue',alpha=color[i])

    myFmt = mdates.DateFormatter('%D')
    ax.xaxis.set_major_formatter(myFmt)
    plt.ylabel('Power [kW]', fontsize=14)
    plt.title(f'{plot_string}', fontsize=14)
    ax.legend()
    plt.savefig(f'{plot_string}_forecast_{str(date.date())}.png', bbox_inches='tight',dpi=(250))
    plt.show()

