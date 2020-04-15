import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from matplotlib import pyplot as plot
from sklearn.preprocessing import MinMaxScaler
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
import statistics as st
import statsmodels.api as sm
from statsmodels.tsa.seasonal import DecomposeResult
import statsmodels.formula.api as smf


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return (self.inputs[i], self.targets[i])


def scale(ts):
    """Scaling data, -1 to 1, ts shape (x, )"""
    scaler_ts = MinMaxScaler(feature_range = (-1, 1))
    scaler_ts = scaler_ts.fit(ts.values.reshape(-1, 1))
    ts_scaled = scaler_ts.transform(ts.values.reshape(-1, 1))
    ts_scaled = pd.DataFrame(ts_scaled)
    d = dict()
    d["scaler"] = scaler_ts
    d["scaled"] = ts_scaled #(x, 1)
    return d


def abline(slope, intercept):
    """Plots a line from slope and intercept"""
    axes = plot.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plot.plot(x_vals, y_vals, '--')


def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = data[i]
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance


def ones_target(size, device):
    data = torch.ones(size, 1)
    data = data.to(device)
    return data


def zeros_target(size, device):
    data = torch.zeros(size, 1)
    data = data.to(device)
    return data


def decompose(df, period=365, lo_frac=0.6, lo_delta=0.01):
    # use some existing pieces of statsmodels
    lowess = sm.nonparametric.lowess
    _pandas_wrapper, _ = _maybe_get_pandas_wrapper_freq(df)

    # get plain np array
    observed = np.asanyarray(df).squeeze()

    # calc trend, remove from observation
    trend = lowess(observed, [x for x in range(len(observed))],
                   frac=lo_frac,
                   delta=lo_delta * len(observed),
                   return_sorted=False)
    detrended = observed - trend

    # period must not be larger than size of series to avoid introducing NaNs
    period = min(period, len(observed))

    # calc one-period seasonality, remove tiled array from detrended
    period_averages = np.array([pd_nanmean(detrended[i::period]) for i in range(period)])
    # 0-center the period avgs
    period_averages -= np.mean(period_averages)
    seasonal = np.tile(period_averages, len(observed) // period + 1)[:len(observed)]
    resid = detrended - seasonal

    # convert the arrays back to appropriate dataframes, stuff them back into
    #  the statsmodel object
    results = list(map(_pandas_wrapper, [seasonal, trend, resid, observed]))
    dr = DecomposeResult(seasonal=results[0],
                         trend=results[1],
                         resid=results[2],
                         observed=results[3],
                         weights=period_averages)
    return dr


def anchor(signal, weight):
    """Data Smooothing"""
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


def strongest_trend_period(ts, start, end):
    """Trend Detection, ts shape(x, )"""
    trend_strengths = []
    for i in range(start, end):
        decomposition = decompose(ts, period=i)
        var_residual = np.power(st.stdev(decomposition.resid), 2)
        var_trend_residual = np.power(st.stdev(anchor(decomposition.trend, 0.9) + decomposition.resid), 2)
        trend_strength = max(0, (1 - var_residual/var_trend_residual))
        trend_strengths.append(trend_strength)
    d = dict()
    d["index"] = trend_strengths.index(max(trend_strengths))
    d["period"] = d["index"] + 1
    d["trend_strength"] = max(trend_strengths)
    return d


def remove_trend(ts, p):
    """Remove Trend, ts shape (x, )"""
    decomposition = decompose(ts, period = p)
    detrended = ts - decomposition.trend
    d = dict()
    d["detrended"] = detrended.reshape(detrended.shape[0], 1)
    d["trend"] = decomposition.trend.reshape(decomposition.trend.shape[0], 1)
    return d


def linearity_score(ts):
    """Linarity Score Detection"""
    x = [e for e in range(0, ts.shape[0])]
    x = np.array(x)
    trend_train_df = pd.DataFrame({'time': x, 'trend':ts})
    model_trend_train = smf.ols('time ~ trend', data = trend_train_df).fit()
    return model_trend_train.rsquared


def strongest_seasonal_period(ts, start, end):
    """Period Detecion, tx shape (x, )"""
    seasonal_strengths = []
    for i in range(start, end):
        decomposition = decompose(ts, period = i)
        var_residual = np.power(st.stdev(decomposition.resid), 2)
        var_seasonal_residual = np.power(st.stdev(decomposition.seasonal + decomposition.resid), 2)
        seasonality_strength = max(0, (1 - var_residual/var_seasonal_residual))
        seasonal_strengths.append(seasonality_strength)
    d = dict()
    d["index"] = seasonal_strengths.index(max(seasonal_strengths))
    d["period"] = d["index"] + 1
    d["seasonality_strength"] = max(seasonal_strengths)
    return d
