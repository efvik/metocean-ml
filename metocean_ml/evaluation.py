from spectra_tools import integrated_parameters
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
import pandas as pd
import numpy as np

def error_metrics_1D(target,pred):
    return {
        "mse": mean_squared_error(target, pred),
        "mae": mean_absolute_error(target, pred),
        "r2": r2_score(target, pred),
        "rmse":root_mean_squared_error(target,pred)}

def spec_prediction_performance(spec_pred, spec_target, freq, dir,upsample=1000):
    def dir_error(target,pred):
        diff = np.abs(target-pred)%360
        diff = np.minimum(diff,360-diff)
        return error_metrics_1D(np.zeros_like(diff),diff)

    params_pred = integrated_parameters(spec_pred,freq,dir,upsample=upsample)
    params_target = integrated_parameters(spec_target,freq,dir,upsample=upsample)
    perf = {}
    for k,pred in params_pred.items():
        target = params_target[k]
        if "dir" in k:
            perf[k] = dir_error(target,pred)
        else:
            perf[k] = error_metrics_1D(target,pred)
    return params_pred, params_target, pd.DataFrame(perf)

