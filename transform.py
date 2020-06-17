""" Functions that can be called as `transform`
"""
import itertools
import numpy as np
import netCDF4 as nc
# import xarray as xr
import pandas as pd
# import datetime


__all__ = ['daily_min', 'daily_max', 'daily_mean', 'monthly_mean', 'yearly_mean', 
    'threshold_negative', 'threshold_positive', 'monthly_count', 'interpolate_daily']

def aggregate_datetime(dates, values, func, field):
    " field is a date(time) field"
    date_val = zip(dates, values)

    date_val2 = []
    for _, group in itertools.groupby(date_val, key=lambda x: getattr(x[0], field)):
        dates, values = zip(*group)
        val = func(values)
        d0 = dates[len(dates)//2] # take middle date
        date_val2.append([d0, val])

    dates2, values2 = zip(*date_val2)

    return dates2, values2

def _aggregate_timeseries(timeseries, func, field):
    if not isinstance(timeseries, pd.Series):
        raise ValueError(f'Expected pandas Series, got {type(timeseries)}')

    dates = nc.num2date(timeseries.index.values, timeseries.index.name)
    dates2, values2 = aggregate_datetime(dates, timeseries.values, func, field)
    index = pd.Index(nc.date2num(dates2, timeseries.index.name))
    index.name = timeseries.index.name
    return pd.Series(values2, index=index)


def daily_min(timeseries):
    return _aggregate_timeseries(timeseries, np.min, 'day')

def daily_max(timeseries):
    return _aggregate_timeseries(timeseries, np.max, 'day')

def daily_mean(timeseries):
    return _aggregate_timeseries(timeseries, np.mean, 'day')

def monthly_mean(timeseries):
    return _aggregate_timeseries(timeseries, np.mean, 'month')

def yearly_mean(timeseries):
    return _aggregate_timeseries(timeseries, np.mean, 'year')

def threshold_negative(timeseries):
    return timeseries <= 0

def threshold_positive(timeseries):
    return timeseries > 0

def monthly_count(timeseries):
    return _aggregate_timeseries(timeseries, lambda t: np.sum(t), 'month')

def interpolate_daily(timeseries):
    if not isinstance(timeseries, pd.Series):
        raise ValueError(f'Expected pandas Series, got {type(timeseries)}')
    day1, dayend = timeseries.index.values[0], timeseries.index.values[-1]
    days = np.arange(day1, dayend+1)
    daily_values = np.interp(days, timeseries.index.values, timeseries.values)
    index = pd.Index(days)
    index.name = timeseries.index.name
    return pd.Series(daily_values, index=index, name=timeseries.name)
