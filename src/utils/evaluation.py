"""
Collection of utility functions used in the model evaluation pipeline.
"""

from functools import partial
from typing import Literal, overload

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xarray as xr
import xskillscore as xs

from src.utils.data_prep import read_prio_training_data
from prediction_competition_2023.IgnoranceScore import ensemble_ignorance_score_xskillscore
from prediction_competition_2023.IntervalScore import mean_interval_score_xskillscore


def create_poisson_benchmark(observed: xr.DataArray, fp_views: str, year: int) -> xr.DataArray:
    """
    Own implementation of last observation poisson benchmarks as ViEWS original data contained
    errors (adapted from Martin's tests)
    
    Loads the training data for the prediciton window, selects the last observations, and 
    draws poisson samples of size N=1000 with lambda corresponding to the observed fatalities.
    
    Args:
        observed (xr.DataArray): Observations, used to copy the metadata.
        fp_views (str): filepath for views data
        year (int): yearly prediction window create the benchmark for.
        
    Return:
        xr.DataArray containing benchmark predictions
    """
    df_train, _ = read_prio_training_data(fp_views, prediction_year=year)
    # get last observations
    last_obs = df_train.loc[df_train.index.get_level_values("month_id").max()]["ged_sb"]
    # fix a seed for reproducibility
    rng = np.random.default_rng(14653221687987913213548546516751591768837241)
    # build as np array rather than list for easy use with xarray
    predictions_poisson_np = np.zeros(shape=(observed.shape[0], observed.shape[1], 1000))
    draws = predictions_poisson_np.shape[2]
    for i in range(len(observed.month_id.values)):
        for j in range(len(last_obs)):
            predictions_poisson_np[i, j] = rng.poisson(last_obs.iloc[j], draws)
    # create DataArray
    predictions_poisson = xr.DataArray(
        data=predictions_poisson_np,
        coords={
            "month_id": observed.coords["month_id"],
            "priogrid_gid": observed.coords["priogrid_gid"],
            "draw": np.arange(1000),
        },
        name="outcome",
    )
    return predictions_poisson


def build_map_estimates(predictions: xr.DataArray) -> xr.DataArray:
    """Calculates MAP estimates for prediction samples.

    Estimates are the max of the kernel density function estimated via gaussian_kde.
    For our predictions size, this takes 2-3 min per year.

    Args:
        predictions (xr.DataArray): yearly predictions

    Returns:
        xr.DataArray with dims month_id, priogrid_gid containing the MAP estimates.
    """

    def draw_func(draws):
        if all(draws == 0):
            return 0
        kde = gaussian_kde(draws)
        possible_vals = np.arange(draws.min(), draws.max() + 1, 1)
        density_values = kde(possible_vals)
        peak = possible_vals[np.argmax(density_values)]
        return peak

    map_estimates = xr.apply_ufunc(
        draw_func,
        predictions,
        input_core_dims=[["draw"]],
        vectorize=True,
    )
    return map_estimates


def calculate_metrics(
    observed: xr.DataArray, predictions: xr.DataArray, name: str = "model"
) -> pd.Series:
    """
    Function to calculate a number of metrics for a prediction: CRPS, Ignorance Score and Mean 
    Interval Scores as defined by ViEWS, additionally MSE and MAE for the MAP estimates based 
    on the individual draws.

    Args:
        observed (xr.DataArray): observed data as basis to calculate metrics.
        predictions (xr.DataArray): predicted data to calculate metrics for.
        name (str, optional): name to give to the output Series with the metrics.

    Returns:
        pd.Series with name "name" and the different metrics as index.
    """
    # the metrics we calculate
    metrics = {
        "crps": partial(xs.crps_ensemble, member_dim="draw"),
        "ign": partial(
            ensemble_ignorance_score_xskillscore,
            bins=[0, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
            member_dim="draw",
        ),
        "mis": partial(
            mean_interval_score_xskillscore,
            member_dim="draw",
            prediction_interval_level=0.9,
        ),
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
    }

    series_metrics = pd.Series(index=pd.Index(metrics.keys(), name="metrics"), name=name)
    map_estimates = build_map_estimates(predictions)
    for metric in metrics:
        if metric in [
            "mse",
            "mae",
        ]:  # For single value metrics use the mean of all draws from the distribution
            performance = metrics[metric](observed, map_estimates)
        else:
            performance = metrics[metric](observed, predictions).values
        series_metrics.at[metric] = performance
    series_metrics = series_metrics.astype(float)
    return series_metrics

@overload
def create_metrics_df(
    observed: xr.DataArray,
    predictions: list[xr.DataArray] | dict[str, xr.DataArray],
    return_results: Literal[True] = True,
) -> pd.DataFrame:
    ...
    
@overload
def create_metrics_df(
    observed: xr.DataArray,
    predictions: list[xr.DataArray] | dict[str, xr.DataArray],
    return_results: Literal[False],
) -> pd.DataFrame:
    ...

def create_metrics_df(
    observed: xr.DataArray,
    predictions: list[xr.DataArray] | dict[str, xr.DataArray],
    return_results: bool = True,
) -> pd.DataFrame | None:
    """
    Function to create a dataframe with metrics calculated for a number of different predictions.

    Args:
        observed (xr.DataArray): observed data as basis to calculate metrics.
        predictions (list[xr.DataArray] | dict[str, xr.DataArray]): predicted data to calculate 
            metrics for. Can be list of DataArrays or dict with names of the
            different predictions as keys - which will then be used as column names.
        return_results (bool, optional): whether to return the resulting dataframe - if not it will 
            simply be printed.

    Returns:
        pd.DataFrame with different metrics as index and different predictions as columns and the 
        corresponding values if return_results=True.
    """

    series_list = []
    for i, preds in enumerate(predictions):
        if type(predictions) is dict:
            series_list.append(calculate_metrics(observed, predictions[preds], preds))  # type: ignore
        else:
            series_list.append(calculate_metrics(observed, preds, f"model{i}"))  # type: ignore
    df_metrics = pd.concat(series_list, axis=1)

    if return_results:
        return df_metrics
    else:
        print("metrics:")
        print(df_metrics)
