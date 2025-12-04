"""
Wrapper for NGBRegressor from the ngboost package (https://github.com/stanfordmlgroup/ngboost) to make it compatible
with our architecture.
"""

from typing import Literal
import warnings

from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal, Exponential
from ngboost.scores import LogScore, CRPScore
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, clone, check_is_fitted


class NGBRegressorWrapper(BaseEstimator):
    """
    Simple sklearn estimator wrapping the NGBRegressor to allows us to treat the DecisionTreeRegressor kwargs as
    hyperparameters during model tuning with our existing architecture.

    'base_' variables are kwargs of the base estimator, all others are simply the kwargs of the underlying NGBRegressor.
    """

    def __init__(
        self,
        base_criterion: Literal[
            "squared_error", "friedman_mse", "absolute_error", "poisson"
        ] = "friedman_mse",
        base_splitter: Literal["best", "random"] = "random",
        base_max_depth: int = 3,
        dist: str = "LogNormal",
        score: str = "CRPScore",
        natural_gradient=True,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        col_sample: float = 1.0,
        verbose: bool = False,
        verbose_eval: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        validation_fraction: float = 0.1,
        early_stopping_rounds: int | None = None,
    ):
        self.base_criterion = base_criterion
        self.base_splitter = base_splitter
        self.base_max_depth = base_max_depth
        self.dist = dist
        self.score = score
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.tol = tol
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y):
        """create base estimator based on base_ params and use that to create NGBRegressor"""
        base_estimator = DecisionTreeRegressor(
            criterion=self.base_criterion,
            splitter=self.base_splitter,  # type: ignore
            max_depth=self.base_max_depth,
            random_state=self.random_state,
        )
        dists = {"LogNormal": LogNormal, "Normal": Normal, "Exponential": Exponential}
        scores = {"CRPScore": CRPScore, "LogScore": LogScore}
        ngb = NGBRegressor(
            Dist=dists[self.dist],
            Score=scores[self.score],
            Base=base_estimator,
            natural_gradient=self.natural_gradient,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            tol=self.tol,
            random_state=self.random_state,
            validation_fraction=self.validation_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.estimator_ = clone(ngb)
        # tends to get runtime warnings from CRPS scorer that are quite annoying
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.estimator_.fit(X, y)
        self.is_fitted_ = True

    def predict(self, X, sample_size=1000):
        """implements the pred_dist prediction method needed for our problem"""
        check_is_fitted(self, "is_fitted_")
        y_hat = self.estimator_.pred_dist(X)
        y_hat = np.array([y_hat.rvs() for i in range(sample_size)]).T.round(0)
        return y_hat
