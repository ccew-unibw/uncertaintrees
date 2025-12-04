"""
Class to handle to global-local ensemble combination.
"""

from collections.abc import Callable
from functools import partial
from typing import Literal, overload

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import xarray as xr
import xskillscore as xs


class GlobalLocalHurdleEnsemble(BaseEstimator):
    """
    "Estimator" class to handle the global-local ensemble combination.

    Args:
        hurdle_method (str): how to merge predictions with zeroes
            "shares": classification probability is treated as fixed share of the 1000 samples to 
                draw from the non-zero regression predictions
            "probabilistic": all regression predictions are probabilistically replaces with 0 based 
                on the classification probability
        clf_calibrate (bool): whether to use the calibrated classification probability or not - 
            requires columns with already calibrated classification predictions during fit
        reg_calibrate (bool): whether to calibrate the regression output or not
        rng_seed (int): random seed for the numpy default Random Number Generator used for drawing 
            from the different levels

    """
    # inheriting from BaseEstimator for access to set_params method
    def __init__(
        self,
        hurdle_method: str = "shares",
        clf_calibrate: bool = False,
        reg_calibrate: bool = True,
        rng_seed: int = 6541327914357521695489979424745645677111304,
    ):
        self.clf_calibrate = clf_calibrate
        self.reg_calibrate = reg_calibrate
        self.rng = np.random.default_rng(rng_seed)

        # input check
        try:
            assert hurdle_method in ["probabilistic", "shares"]
            self.hurdle_method = hurdle_method
        except AssertionError:
            raise ValueError(
                f'Argument hurdle_method must be one of ["probabilistic", "shares"]. Got {hurdle_method}.'
            )

    def fit(
        self,
        predictions: pd.DataFrame,
        observations: pd.DataFrame,
        scorer: Callable | None = None,
    ):
        """
        Fit tries possible global-local combinations of the raw predictions regarding their
        performance and stores the resulting combination structure in the ensemble_params_
        attribute. In a prediction setting this generally needs to be done with known, i.e.
        older, calibration data. Optionally also performs a simple cluster-wise regression
        calibration based on the mean number of fatalities.

        Args:
            predictions (pd.DataFrame): dataframe with raw calibration predictions
            observations (pd.DataFrame): dataframe with calibration true values
            scorer (Callable | None, optional): scorer to judge the combinations of raw predictions 
                (currently takes two xarrays as arguments). Defaults to CRPS if None
        """
        # default scorer
        if scorer is None:
            scorer = partial(xs.crps_ensemble, member_dim="draw")
        # indices need to match
        assert predictions.index.equals(observations.index)
        # column checks
        for col in [
            "classification",
            "classification_local",
            "regression",
            "regression_local",
        ]:
            assert col in predictions.columns
        if self.clf_calibrate:
            for col in ["classification_calibrated", "classification_local_calibrated"]:
                assert col in predictions.columns

        self.clusters_ = sorted(predictions.cluster.unique())
        self.columns_ = {
            "classification": {
                "global": "classification"
                if not self.clf_calibrate
                else "classification_calibrated",
                "local": "classification_local"
                if not self.clf_calibrate
                else "classification_local_calibrated",
            },
            "regression": {"global": "regression", "local": "regression_local"},
        }
        ensemble_params = dict.fromkeys(self.clusters_)
        for c in self.clusters_:
            cluster_index = predictions[predictions.cluster == c].index
            predictions_cluster = predictions.loc[cluster_index]
            observations_cluster = observations.loc[cluster_index]
            # calculate scores and select best global-local combinations per cluster
            scores = pd.DataFrame(
                index=["classification_global", "classification_local"],
                columns=["regression_global", "regression_local"],
            ).astype(float)
            # scoring matrix
            for clf in ["global", "local"]:
                for reg in ["global", "local"]:
                    scores.at[f"classification_{clf}", f"regression_{reg}"] = self.combine_levels(
                        predictions_cluster,
                        clf,
                        reg,
                        scorer=scorer,
                        actuals=observations_cluster["ged_sb"],
                    )

            best: tuple[str, str] = scores.unstack().idxmin()
            ensemble_params[c] = dict.fromkeys(["classification", "regression"])
            # save best global/local configuration to ensemble_params
            for step in best:
                ensemble_params[c][step[: step.find("_")]] = step[step.find("_") + 1 :]

            if self.reg_calibrate:
                # calculate multiplier for outcome calibration
                outcome = self.combine_levels(
                    predictions_cluster,
                    ensemble_params[c]["classification"],
                    ensemble_params[c]["regression"],
                    result="data",
                )

                cluster_preds_mean = outcome.apply(lambda x: x.mean()).mean()
                cluster_obs_mean = observations_cluster.ged_sb.mean()
                multiplier = cluster_obs_mean / cluster_preds_mean
                ensemble_params[c]["calibration_multiplier"] = multiplier

        self.ensemble_params_ = ensemble_params

    def predict(self, predictions_raw: pd.DataFrame) -> xr.DataArray:
        """
        Performs the ensemble combination based on the choices stored in ensemble_params_
        variable generated during fit.

        Args:
            predictions_raw (pd.DataFrame): dataframe with raw predictions

        Returns:
            combined predictions as xarray DataArray
        """
        check_is_fitted(self, ["columns_", "clusters_", "ensemble_params_"])
        # input checks
        required_columns = (
            [c for c in self.columns_["classification"].values()]
            + [c for c in self.columns_["regression"].values()]
            + ["cluster"]
        )
        for c in required_columns:
            try:
                assert c in predictions_raw.columns
            except AssertionError:
                raise ValueError(
                    f"Column {c} is required for ensembling but was not found in input dataframe."
                )

        try:
            # not the most comprehensive check, but at least the number and names need to match
            assert self.clusters_ == sorted(predictions_raw.cluster.unique())
        except AssertionError:
            raise ValueError(
                "Clusters in input dataframe need to match the clusters seen during fit."
            )

        df_predictions = pd.DataFrame(index=predictions_raw.index, columns=["outcome"])

        # "predict"(combine) on a per-cluster basis according to the ensemble_params
        for c in self.clusters_:
            cluster_predictions = predictions_raw.loc[predictions_raw.cluster == c]
            classification_level = self.ensemble_params_[c]["classification"]
            regression_level = self.ensemble_params_[c]["regression"]
            outcome = self.combine_levels(
                cluster_predictions,
                classification_level,
                regression_level,
                result="data",
            )
            if self.reg_calibrate:
                outcome = outcome.apply(
                    lambda x: np.round(x * self.ensemble_params_[c]["calibration_multiplier"])
                )
            df_predictions.loc[outcome.index, "outcome"] = outcome

        da_predictions = self.convert_sample_col_to_xarray_(df_predictions.outcome)

        return da_predictions

    def hurdle_predictions_(
        self, p: float, draws: np.ndarray, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """
        creates combined draw from regression sample (draws) and all-zero sample based on
        classification probability (p). optionally takes a np.random.Generator or uses
        default generator from class instance. sample size depends on the size of the passed
        sample

        Args:
            p (float): classification probability determining zero share
            draws (np.ndarray): regression draws to mix with zeroes
            rng (np.random.Generator | None): optional RNG to use for sampling
        """
        if rng is None:
            # initialize default random number generator
            rng = self.rng
        sample_size = len(draws)
        if self.hurdle_method == "probabilistic":
            # keep values with probability p else replace with 0
            mask = rng.uniform(size=sample_size) > p
            hurdle_draws = draws.copy()
            hurdle_draws[mask] = 0
        elif self.hurdle_method == "shares":
            # fixed number of samples drawn from non-zero based on p
            n_non_zero = round(p * sample_size)
            n_zero = sample_size - n_non_zero
            zeroes = np.zeros(n_zero)
            non_zeroes = rng.choice(draws, size=n_non_zero, replace=False)
            hurdle_draws = np.concatenate([zeroes, non_zeroes])
        else:
            raise ValueError()  # can never happen
        return hurdle_draws.astype(int)
    
    @overload
    def combine_levels(
        self,
        predictions_raw: pd.DataFrame,
        clf_col: str,
        reg_col: str,
        fitted: bool = True,
        *,
        result: Literal["data"],
        scorer: Callable = xs.crps_ensemble,
        actuals: pd.DataFrame | None = None,
    ) -> pd.Series:
        ...

    @overload
    def combine_levels(
        self,
        predictions_raw: pd.DataFrame,
        clf_col: str,
        reg_col: str,
        fitted: bool = True,
        *,
        result: Literal["score"] = "score",
        scorer: Callable = xs.crps_ensemble,
        actuals: pd.DataFrame | None = None,
    ) -> float:
        ...

    def combine_levels(
        self,
        predictions_raw: pd.DataFrame,
        clf_col: str,
        reg_col: str,
        fitted: bool = True,
        *,
        result: Literal["score", "data"] = "score",
        scorer: Callable = xs.crps_ensemble,
        actuals: pd.DataFrame | None = None,
    ) -> float | pd.Series:
        """
        combine the selected levels from the raw predictions to our hurdle predictions and
        return either specified metric score or combined data

        Args:
            predictions_raw (pd.DataFrame): dataframe with raw predictions to combine
            clf_col (str): level ('global'/'local') for classification predictions if
                fitted=True or column name to use if fitted=False
            reg_col (str): level ('global'/'local') for regression predictions if
                fitted=True or column name to use if fitted=False
            fitted (bool): boolean determining whether clf_col and reg_col are used to choose
                column names from fitted class instance or whether they need to be provided
                specifically
            result (Literal["score", "data"]): whether to return the score ('score')
                returned by the scorer or the combined outcome as a dataframe ('data')
            scorer (Callable): scorer to calculate the score based on the outcome generated
                and the passed actuals. only required if result == 'score'
            actuals (pd.DataFrame | None): dataframe with actuals to calculate the score.
                only required if result == 'score'

        Returns:
            score (float) from the scorer or pd.DataFrame with combined predictions,
            depending on the 'result' argument

        """
        if result == "score":
            assert actuals is not None

        if fitted:
            clf_col = self.columns_["classification"][clf_col]
            reg_col = self.columns_["regression"][reg_col]

        outcome = predictions_raw.apply(
            lambda x: self.hurdle_predictions_(x[clf_col], x[reg_col]), axis=1
        )
        if result == "score":
            score = float(scorer(actuals.to_xarray(), self.convert_sample_col_to_xarray_(outcome)))  # type: ignore
            return score
        elif result == "data":
            outcome.name = "draw"
            return outcome
        else:
            raise ValueError

    def convert_sample_col_to_xarray_(self, series: pd.Series) -> xr.DataArray:
        """
        Converts a pandas series with sample predictions stored as arrays with len=1000 for
        each pgm to xarray, mostly for calculating scores.
        """
        series = series.sort_index()  # just to be safe
        # need to do a bit of pandas/numpy magic to reshape to 3d array
        data = series.unstack().values
        data_3d = np.zeros((data.shape[0], data.shape[1], 1000))
        for x in range(data.shape[0]):
            data_3d[x] = np.stack(data[x])

        da_predictions = xr.DataArray(
            data=data_3d,
            coords={
                "month_id": series.index.get_level_values(0).unique(),
                series.index.names[1]: series.index.get_level_values(1).unique(),
                "draw": np.arange(1000),
            },
            name="outcome",
        )
        return da_predictions
