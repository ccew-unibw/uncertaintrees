"""
Utility functions used for prediction and scoring, including the base scorer and prediction
wrappers used to create the scorers for hyperparameter tuning in the regression pipeline.
"""

from collections.abc import Callable

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
import xarray as xr
import xskillscore as xs

from src.estimators.drf import DistributionalRandomForestRegressor
from src.estimators.ngboost import NGBRegressorWrapper

# tested but discared for now
# def ngb_fix_drawing(predictions):
#     predictions = predictions.copy()
#     for i in range(predictions.shape[0]):
#         rng = np.random.default_rng(651654687446165143541688134131154)
#         non_zeros = predictions[i][predictions[i] != 0]
#         replacement = rng.choice(non_zeros, size=1000-len(non_zeros))
#         predictions[i] = np.concatenate([non_zeros, replacement])
#     return predictions


def qrf_predict(qrf: RandomForestQuantileRegressor, X) -> np.ndarray:
    """Wrapper function to ensure consistent prediction output of shape(len(X), 1000) for 
    Quantile Random Forest
    """
    y_hat = qrf.predict(
        X.values, quantiles=list(np.linspace(0, 1, 1000)), interpolation="linear"
    ).round(0)
    return y_hat


def drf_predict(drf: DistributionalRandomForestRegressor, X) -> np.ndarray:
    """Wrapper function to ensure consistent prediction output of shape(len(X), 1000) for 
    Distributional Random Forest
    """
    y_hat = drf.predict(X, functional="sample", n_samples=1000)
    y_hat = y_hat.squeeze() # type: ignore
    return y_hat


def ngb_predict(ngb: NGBRegressorWrapper, X) -> np.ndarray:
    """Wrapper function to ensure consistent prediction output of shape(len(X), 1000) for NGBoost
    """
    y_hat = ngb.predict(X)
    y_hat[y_hat < 1] = 1  # predicted output includes 0 - simply replace with 1 for now
    return y_hat


def calculate_crps_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """builds DataArrays from y and y_hat and returns CRPS score
    
    Args:
        y (np.ndarray): observations (1D)
        y_hat (np.ndarray): predictions (2D, samples from distributional output)
        
    Return:
        float: mean CRPS score
    """
    predictions = xr.DataArray(
        data=y_hat,
        coords={
            "id": range(len(y_hat)),
            "member": np.arange(y_hat.shape[1]),
        },
        name="predictions",
    )
    observed = xr.DataArray(
        data=y,
        coords={
            "id": range(len(y)),
        },
        name="predictions",
    )
    score = xs.crps_ensemble(observed, predictions).values
    return float(score)


def crps_scorer(estimator, X, y, func: Callable) -> float:
    """
    Base function to create the different scorers for the regressors. Designed to be used with a
    function handling how the predictions are generated frozen with functools.partial to deal
    with different types of estimators, as sklearn scorers need to have the signature
    scorer(estimator, X, y).

    Args:
        estimator: fitted estimator to score.
        X: data with features.
        y: data with observations.
        func (Callable): function generating predictions of the estimator given X. Called with
            func(estimator, X).

    Returns:
        CRPS score for estimator's predictions from X compared to y
    """
    # data check
    try:
        assert len(X) == len(y)
    except AssertionError:
        raise ValueError("X and y are not of same length. Scoring therefore not possible.")

    y_hat = func(estimator, X)
    try:
        # prediction format check - len 1 (usually caused by calling squeeze()) will fail during calculate_crps_score,
        # but is also not informative, so not fixing this - some clusters are a bit sparse for the cross-validation
        assert len(y_hat.shape) == 2
    except AssertionError:
        print("y shape:", y.shape, "y_hat shape:", y_hat.shape)
        raise
    score = calculate_crps_score(y, y_hat)
    # we return negative values so greater is better for consistency with classifier scores
    return -score
