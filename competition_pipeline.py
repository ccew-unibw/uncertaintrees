#!/usr/bin/env python3.10
"""
Main pipeline logic. Tunes & trains models, generates predictions and stores
output. All config for the predictions is handled here.
"""

from collections.abc import Callable
from functools import partial
import argparse
import os
import pickle
import time
from typing import Any, Literal

from hyperopt import hp
import ml_insights as mli
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, make_scorer
import xskillscore as xs

from src.utils.conversion import get_date
from src.utils.data_prep import (
    read_prio_training_data,
    read_prio_actuals,
    read_raw_predictions,
    level_index,
)
from src.estimators.hurdle import HurdleStepsWithTuning
from src.estimators.ensemble import GlobalLocalHurdleEnsemble
from src.utils.scoring import drf_predict, qrf_predict, ngb_predict, crps_scorer
from src.utils.clustering import (
    make_clusters_hdbscan,
    make_clusters_unregions,
    make_clusters_dbscan,
    make_clusters_test_pgm,
)


def create_clusters(
    visualize_clusters: bool = False, regenerate_clusters: bool = False
) -> pd.DataFrame:
    """
    Function to create and store / load possible cluster assignments.

    Relies on config via global variables:
        clusters (str): defines the clustering method to be used
        fp_views (str): filepath for views data
        level (str): whether to run country or grid predictions

    Args:
        visualize_clusters (bool): whether to also plot a graphic with the clusters
        regenerate_clusters (bool): whether to ignore existing cluster assignments

    Returns:
        pandas DataFrame with cluster assignments of the specified method

    """

    # currently only 1 full cluster
    if level == "cm":
        df_train, _ = read_prio_training_data(fp_views, level="cm")
        clust_df = pd.Series(
            index=df_train.index.get_level_values(1).unique(), data=0, name="cluster"
        ).to_frame()
        return clust_df

    if test_mode:
        return make_clusters_test_pgm()

    cluster_options = ["hdbscan", "dbscan", "unregions"]
    # input check
    try:
        assert clusters in cluster_options
    except AssertionError:
        raise ValueError(
            f'Argument clusters needs to be one of {cluster_options}. "{clusters}" passed'
        )

    # load prepared clusters_dict if already done
    if os.path.exists("data/clusters_dict.pkl") and not regenerate_clusters:
        print("Loading already created cluster assignments...")
        with open("data/clusters_dict.pkl", "rb") as f:
            clusters_dict = pickle.load(f)
    # create all possible clusters and store
    else:
        print("Creating cluster assignments...")
        # Clustering is done based on the initial training data up to 10-2017
        df_train, _ = read_prio_training_data(fp_views)
        clusters_dict = dict.fromkeys(cluster_options)
        # create clusters
        dbscan_kwargs = {"eps": 1.5, "min_samples": 10}
        clusters_dict["dbscan"] = make_clusters_dbscan(
            df_train,
            cluster_kwargs=dbscan_kwargs,
            visualize_clusters=visualize_clusters,
        )
        hdbscan_kwargs = {
            "min_cluster_size": 18,
            "min_samples": 15,
            "cluster_selection_epsilon": 2.2,
        }
        clusters_dict["hdbscan"] = make_clusters_hdbscan(
            df_train,
            min_months=1000,
            cluster_kwargs=hdbscan_kwargs,
            visualize_clusters=visualize_clusters,
        )
        clusters_dict["unregions"] = make_clusters_unregions(
            df_train, visualize_clusters=visualize_clusters
        )
        with open("data/clusters_dict.pkl", "wb") as f:
            pickle.dump(clusters_dict, f)

    return clusters_dict[clusters]  # type: ignore


def predictions_to_df(preds_clf, preds_reg, target, prediction_month, pgids, local=False):
    """
    Helper function to store predictions from classifier and regressor in single dataframe. This assumes
    predictions in the shape (n_rows, 1, n_samples).
    """
    predictions = pd.DataFrame(index=pgids)
    predictions.index.name = level_index(level)[1]
    month_id = prediction_month + int(target[target.rfind("_t") + 2 :])
    predictions["month_id"] = month_id
    predictions = predictions.set_index("month_id", append=True).reorder_levels([1, 0])
    predictions["classification"] = preds_clf[:, 1]
    predictions["regression"] = [preds_reg[i, :] for i in range(preds_reg.shape[0])]
    if local:
        predictions = predictions.rename(
            columns={
                "classification": "classification_local",
                "regression": "regression_local",
            }
        )
    return predictions


def make_predictions(
    df_train: pd.DataFrame,
    features: list[str],
    target_reg: str,
    target_clf: str,
    fp_tuning: str,
    clust_df: pd.DataFrame,
    ignore_prior_tuning: bool,
    random_state: int,
    calibrate_classifier: bool = False,
    **calibration_kwargs,
) -> pd.DataFrame:
    """
    Makes predictions for the global and the different local models for one prediction timestep.
    Includes both components, classifier and regressor, of the hurdle combinations and returns the
    "raw" predictions. Includes tuning of the hurdle class and a simple version of spline
    calibration for the classifier (optional).

    Args:
        df_train (pd.DataFrame): training data (including targets)
        features (list[str]): list of features to use for both components of the hurdle estimator
        target_reg (str): regression target column name
        target_clf (str): classification target column name
        fp_tuning (str): filepath where tuning results of the hurdle class are stored/loaded from
        clust_df (pd.DataFrame): dataframe with cluster assignments which is used to select the clusters for the
            local models
        ignore_prior_tuning (bool): whether to skip checks for existing tuning results
        random_state (int): random state passed to models
        calibrate_classifier (bool, optional): whether to run the spline calibration
        **calibration_kwargs: kwargs ['calibration_months', 'min_positives'] to be passed to
            calibration function

    Returns:
        pd.DataFrame: "raw" predictions for given prediction timestep
    """

    def spline_calibration(
        df_data: pd.DataFrame,
        hurdle_estimator: HurdleStepsWithTuning,
        df_predictions: pd.DataFrame,
        calibration_months: int = 24,
        min_positives: int = 200,
        local: bool = False,
    ) -> pd.DataFrame:
        print("calibrating classification predictions...")
        # get classifier
        clf = clone(hurdle_estimator.clf)
        # split training data in train and calibration set
        target_max = df_data.index.get_level_values("month_id").max()
        df_train_calib = df_data.loc[: target_max - calibration_months]
        df_test_calib = df_data.loc[target_max - calibration_months :]
        # fit and predict - got an error where the array couldn't be set to writable from sklearn checks without .values
        clf.fit(df_train_calib[features], df_train_calib[target_clf].values)
        preds_calib = clf.predict_proba(df_test_calib[features])[:, 1]
        # match predictions to observations
        df_calib = pd.DataFrame(index=df_test_calib.index, data=preds_calib, columns=["prediction"])
        df_calib["observation"] = df_test_calib[target_clf].values
        # calibrate
        calib_global = mli.SplineCalib()
        calib_global.fit(df_calib.prediction, df_calib.observation)
        # calibrate on a cluster by cluster basis for the global predictions
        if not local:
            df_predictions["classification_calibrated"] = np.nan
            for c in sorted(clust_df.cluster.unique()):
                geo_ids = clust_df[clust_df.cluster == c].index
                # if there are not enough positives in the data it doesn't really work...
                if (
                    sum(df_calib.loc[:, geo_ids, :].observation != 0) < min_positives
                    or level == "cm"
                ):
                    df_predictions.loc[(slice(None), geo_ids), "classification_calibrated"] = (
                        calib_global.calibrate(df_predictions.loc[:, geo_ids, :].classification)
                    )
                else:
                    calib_local = mli.SplineCalib()
                    calib_local.fit(
                        df_calib.loc[:, geo_ids, :].prediction,
                        df_calib.loc[:, geo_ids, :].observation,
                    )
                    df_predictions.loc[(slice(None), geo_ids), "classification_calibrated"] = (
                        calib_local.calibrate(df_predictions.loc[:, geo_ids, :].classification)
                    )
        else:
            df_predictions["classification_local_calibrated"] = calib_global.calibrate(
                df_predictions.classification_local
            )

        return df_predictions

    def get_global_predictions() -> pd.DataFrame:
        """tunes hurdle class and generates global predictions"""
        print("Generate global predictions...")
        X = df_train.loc[prediction_month, features]

        global_hurdle = HurdleStepsWithTuning(
            fp_tuning=fp_tuning, random_state=random_state, **hurdle_kwargs
        )
        global_hurdle.tune(
            df_train_target,
            features,
            target_reg,
            ignore_prior_tuning=ignore_prior_tuning,
        )
        print("refitting to whole dataset...")
        global_hurdle.fit(df_train_target, features, target_reg)
        preds_clf, preds_reg = global_hurdle.predict(X)
        predictions_global = predictions_to_df(
            preds_clf,
            preds_reg,
            target_reg,
            prediction_month,
            X.index.get_level_values(0),
        )
        # calibration
        if calibrate_classifier:
            predictions_global = spline_calibration(
                df_train_target,
                global_hurdle,
                predictions_global,
                local=False,
                **calibration_kwargs,
            )
        return predictions_global.sort_index()

    def parallel_local(c) -> pd.DataFrame:
        """tunes hurdle class and generates local predictions for cluster c"""
        local_pgids = list(clust_df[clust_df.cluster == c].index)
        df_cluster = df_train_target.loc[pd.IndexSlice[:, local_pgids], :]
        X_local = df_train.loc[pd.IndexSlice[prediction_month, local_pgids], features]

        local_hurdle = HurdleStepsWithTuning(
            fp_tuning=fp_tuning, random_state=random_state, **hurdle_kwargs
        )
        local_hurdle.tune(
            df_cluster,
            features,
            target_reg,
            local_model=True,
            cluster=c,
            ignore_prior_tuning=ignore_prior_tuning,
        )
        local_hurdle.fit(df_cluster, features, target_reg)
        preds_clf, preds_reg = local_hurdle.predict(X_local)
        predictions_cluster = predictions_to_df(
            preds_clf, preds_reg, target_reg, prediction_month, local_pgids, local=True
        )

        # calibration
        if calibrate_classifier:
            predictions_cluster = spline_calibration(
                df_cluster, local_hurdle, predictions_cluster, local=True
            )
        return predictions_cluster

    def get_local_predictions() -> pd.DataFrame:
        """wrapper for parallelized local predictions via joblib"""
        print("Generate local predictions...")
        clusters = set(clust_df.cluster)
        # I suspect this parallelization was one step too much and causing issues on our setup
        # pred_dfs = Parallel(n_jobs=3, verbose=1)(
        #     delayed(parallel_local)(c) for c in clusters
        # )
        pred_dfs = [parallel_local(c) for c in clusters]
        predictions_local = pd.concat(pred_dfs)
        return predictions_local.sort_index()

    # drop nan where we don't have targets in training data
    df_train_target = df_train[features + [target_clf, target_reg]].dropna().sort_index()
    prediction_month = df_train.index.get_level_values("month_id").max()

    global_predictions = get_global_predictions()
    if level == "pgm":
        local_predictions = get_local_predictions()
        # sanity check
        assert all(global_predictions.index == local_predictions.index)
        predictions_raw_tx = pd.concat([global_predictions, local_predictions], axis=1)
    else:
        predictions_raw_tx = global_predictions
    return predictions_raw_tx


def prediction_pipeline(
    year: int,
    fp_raw: str,
    years_prior: int = 3,
    cluster_kwargs: dict | None = None,
    regenerate_predictions: bool = False,
    ignore_prior_tuning: bool = False,
    random_state: int = 1,
    calibration_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Function to handle the process logic of individual window raw predictions.
    Pipline consists of clustering, model tuning, fit and predict for one window, for both global 
    and the local models. Note that clusters are regenerated/-loaded here for each window so the 
    training data does not need to be kept in memory outside of this function. The classification 
    and regression predictions are both output raw - the ensemble combination happens afterward.

    NOTE: The "2024" window corresponds to July 2024 to June 2025 in accordance with the ViEWS 
    challenge while other windows reflect Jan-Dec of the specified year.

    Relies on config via global variables:
        fp_views (str): filepath where views data is stored (currently only works with old versions)

    Args:
        year (int): defines the respective prediction window to generate predictions for.
        fp_raw (str): filepath where raw predictions are stored.
        years_prior (int, optional): number of prior years of predictions that should be available 
            for calibration and ensembling. Note that this is only checked for year=2018 so 
            excluding this may lead to unexpected behaviour!
        cluster_kwargs (dict, optional): kwargs for create_clusters function if change from defaults
            is desired
        regenerate_predictions (bool, optional): whether to generate new predictions or check for 
            existing files and return those.
        ignore_prior_tuning (bool, optional): whether to skip checks for existing tuning results
        random_state (int, optional): random state passed to estimators
        calibration_kwargs (dict, optional): kwargs for spline_calibration function if change from 
            defaults is desired

    Returns
        dataframe with all "raw" predictions for the selected prediction window
    """

    def monthly_predictions(df):
        """the iteration through the 12 months in a window is handled here - 
        returns combined predictions
        """
        features = f_dict["features"]
        # for all 12 months in target year
        predictions_list = []
        for i in range(12):
            print(f"## Generating prediction for t{3 + i}... ##")
            # reading already generated months, useful e.g. in case of crashes
            if (
                os.path.exists(
                    os.path.join(fp_raw, f"temp_{year}/predictions_raw_t{3 + i}.parquet")
                )
                and not regenerate_predictions
            ):
                predictions_raw = pd.read_parquet(
                    os.path.join(fp_raw, f"temp_{year}/predictions_raw_t{3 + i}.parquet")
                )
                predictions_list.append(predictions_raw)
            else:
                target_clf = f_dict["targets_clf"][i]
                target_reg = f_dict["targets_reg"][i]
                # make predictions is the workhorse which includes tuning, predicting, and calibration and handles all the
                # global/local models
                predictions_raw = make_predictions(
                    df,
                    features,
                    target_reg,
                    target_clf,
                    fp_tuning,
                    clust_df,
                    ignore_prior_tuning,
                    random_state,
                    **calibration_kwargs, # type: ignore
                )

                # save in between, e.g. in case of crashes
                if not os.path.exists(os.path.join(fp_raw, f"temp_{year}")):
                    os.makedirs(os.path.join(fp_raw, f"temp_{year}"))
                predictions_raw.to_parquet(
                    os.path.join(fp_raw, f"temp_{year}/predictions_raw_t{3 + i}.parquet")
                )

                predictions_list.append(predictions_raw)

        predictions_combined = pd.concat(predictions_list)
        if level == "pgm":
            predictions_combined = (
                pd.merge(
                    predictions_combined.reset_index(),
                    clust_df[["cluster"]],
                    left_on="priogrid_gid",  # type: ignore
                    right_on="priogrid_gid",
                    how="left",
                )
                .set_index(predictions_combined.index.names)
                .sort_index()
            )
        return predictions_combined

    def window_prediction_exists(year: int | str) -> bool:
        return os.path.exists(os.path.join(fp_raw, f"predictions_{year}.parquet"))

    calibration_kwargs, cluster_kwargs = validate_kwargs_arguments(
        calibration_kwargs, cluster_kwargs
    )

    try:
        assert level in ["pgm", "cm"]
    except AssertionError:
        raise ValueError(f'global "level" config needs to be either "pgm" or "cm", got {level}.')

    start_time = time.time()
    print(f"Generating predictions for window {year}.")

    skip_current_window = False
    if window_prediction_exists(year) and not regenerate_predictions:
        print("Loading already generated predictions.")
        predictions_window = pd.read_parquet(os.path.join(fp_raw, f"predictions_{year}.parquet"))

        # since we do extra predictions for calibration within the first window (2018), we need to 
        # check if these have already all been successfully generated before we skip to the next window
        if year == 2018:
            if not all(window_prediction_exists(w) for w in range(year - years_prior, year)):
                # this is one case where we cannot go next, yet, but we still don't want to rerun 
                # the predictions for the current window
                skip_current_window = True
            else:
                return predictions_window
        else:
            return predictions_window

    # still load the data since we need it for the calibration predictions
    fp_tuning = f"modelling/tuning_trials_{clusters}/"

    # Load training data from ViEWS
    df_train, f_dict = read_prio_training_data(
        fp_views, prediction_year=year, level=level, test_data=test_mode
    )
    print(
        "Training data end:",
        get_date(df_train.index.get_level_values("month_id").max()),
    )

    # get cluster assignments
    clust_df = create_clusters(**cluster_kwargs)

    # the if is for the 2018 window case where we may have predictions but still need the calibration predictions
    if not skip_current_window:  # default is False
        # get predictions
        predictions_window = monthly_predictions(df_train)
        # save predictions
        predictions_window.to_parquet(os.path.join(fp_raw, f"predictions_{year}.parquet"))

    print(
        f"Done. Generating predictions for window {year} took {round((time.time() - start_time) / 60, 1)} minutes."
    )
    # if in the first test window, generate predictions for years before the first test window for 
    # ensemble calibration
    # Note: while anything before window 2018 has been seen during tuning and there might be some 
    # data leakage here, those predictions are only used to simulate ensemble tuning for some 
    # windows, so any these issues should be fairly negligible
    if year == 2018:
        print(f"Generate past predictions for hurdle calibration for {years_prior} years prior...")
        # using 2018 as a constant here for readability since it's not dynamic
        for year in range(2018 - years_prior, 2018):
            print(f"## window {year} ##")
            if window_prediction_exists(year) and not regenerate_predictions:
                print("Predictions already generated, continuing...")
            else:
                # drop data from train df as required for the respective "negative window"
                calibration_prediction_month = df_train.index.get_level_values(
                    "month_id"
                ).max() - 12 * (2018 - year)
                df_calibrate = df_train.loc[:calibration_prediction_month]
                print(
                    "Training data end:",
                    get_date(df_calibrate.index.get_level_values("month_id").max()),
                )

                predictions_calibration_window = monthly_predictions(df_calibrate)
                predictions_calibration_window.to_parquet(
                    os.path.join(fp_raw, f"predictions_{year}.parquet")
                )

    return predictions_window # type: ignore


def validate_kwargs_arguments(
    calibration_kwargs: None | dict[str, Any], cluster_kwargs: None | dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    # input validation for calibration_kwargs
    if calibration_kwargs is not None:
        try:
            assert all(
                key in ["calibrate_classifier", "calibration_months", "min_positives"]
                for key in calibration_kwargs.keys()
            )
        except AssertionError:
            raise ValueError(
                f"All keys passed in calibration_kwargs need to be in ['calibrate_classifier', "
                f"'calibration_months', 'min_positives']. Got {calibration_kwargs.keys()} instead."
            )
    else:
        calibration_kwargs = {}
    # input validation for cluster_kwargs
    if cluster_kwargs is not None:
        try:
            assert all(
                key in ["visualize_clusters", "regenerate_clusters"]
                for key in cluster_kwargs.keys()
            )
        except AssertionError:
            raise ValueError(
                f"All keys passed in cluster_kwargs need to be in ['visualize_clusters', "
                f"'regenerate_clusters']. Got {cluster_kwargs.keys()} instead."
            )
    else:
        cluster_kwargs = {}

    return calibration_kwargs, cluster_kwargs


def ensemble_predictions(year: int) -> None:
    """
    Runs the prediction_pipeline to get raw predictions and generates ensemble predictions for a 
    given year. Saves combined predictions.

    NOTE: The "2024" window corresponds to July 2024 to June 2025 in accordance with the ViEWS 
    challenge while other windows reflect Jan-Dec of the specified year.

    Relies on config via global variables:
        fp_views (str): filepath where views data is stored
        clusters (str): clustering method - only applies to pgm level
        model_name (str): determines filepath to store the final submission
        level (str): whether to run country or grid predictions
        prediction_kwargs (dict): kwargs passed to prediction_pipeline
            Possible prediction_kwargs are
            - years_prior
            - cluster_kwargs
            - regenerate_predictions
            - ignore_prior_tuning
            - random_state
            - test_mode
            - calibration_kwargs
            For more details see prediction_pipeline docstring.

    Args:
        year (int): year of the prediction window

    """
    if "prediction_kwargs" not in globals():
        global prediction_kwargs
        prediction_kwargs = {}

    fp_raw = f"modelling/raw_predictions_{clusters}"  # path to store raw predictions

    print("Mode:", f"{'TEST' if test_mode else 'REALITY'}")
    print(f"##### window: {year}, level: {level} #####")
    predictions_raw = prediction_pipeline(year=year, fp_raw=fp_raw, **prediction_kwargs)

    print("Creating combined predictions: simple combinations...")
    ensemble = GlobalLocalHurdleEnsemble(
        clf_calibrate=False,
        reg_calibrate=False,
    )
    # global col names
    predictions_global = ensemble.combine_levels(
        predictions_raw, "classification", "regression", fitted=False, result="data"
    )
    predictions_dict = {
        "global": predictions_global,
    }
    # no "local" cm models
    if level == "pgm":
        # local col names
        predictions_local = ensemble.combine_levels(
            predictions_raw,
            "classification_local",
            "regression_local",
            fitted=False,
            result="data",
        )

        predictions_dict["local"] = predictions_local

    # The xarray conversion via numpy and later .to_dataframe() are significantly more efficient than pandas explode
    predictions_dict = {
        key: ensemble.convert_sample_col_to_xarray_(value)
        for key, value in predictions_dict.items()
    }

    # no "local" cm models
    # combination relies on past performance, which does not exist in test mode
    if level == "pgm" and not test_mode:
        print("Creating combined predictions: global-local combination...")
        print("load calibration data...")
        if "years_prior" in prediction_kwargs.keys():
            years_prior = prediction_kwargs["years_prior"]
        else:
            years_prior = 3  # defaults to the same as in prediction_pipeline
        load_data_years = np.arange(
            year - years_prior, year
        )  # 3 years of calibration data for ensemble
        predictions_calibration, observed_calibration = read_raw_predictions(
            fp_views, fp_raw, load_data_years
        )

        print("global-local combination...")
        ensemble.fit(predictions_calibration, observed_calibration)
        print("global-local parameters:\n", ensemble.ensemble_params_)
        predictions_dict["global-local"] = ensemble.predict(predictions_raw)

    if year != 2024:
        # calculate and print the main metric as a bit of initial information
        observed = read_prio_actuals(fp_views, year=year, level=level)
        for model in predictions_dict:
            print(
                model,
                f"{year} crps:",
                xs.crps_ensemble(observed, predictions_dict[model], member_dim="draw").values,
            )

    print("saving predictions...")
    for model in predictions_dict:
        fp_out = f"submissions/unibw_trees_{model}/"
        if test_mode:
            fp_out = f"submissions/test_{model}/"
        if not os.path.exists(fp_out):
            os.makedirs(fp_out)
        # save - store with index as columns to satisfy submission requirements

        df_out = predictions_dict[model].to_dataframe()

        fp_out_pred = os.path.join(fp_out, level, f"window=Y{year}")
        if not os.path.exists(fp_out_pred):
            os.makedirs(fp_out_pred)
        df_out.to_parquet(
            os.path.join(fp_out_pred, f"unibw_trees_{model}_predictions_{year}.parquet")
        )
    print("...done.")
    return


if __name__ == "__main__":
    ###############################################
    ### config and tuning spaces for our models ###
    ###############################################

    # Very basic CLI argument parsing for the test mode
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--test", help="run the pipeline in test mode", action="store_true"
    )
    args = parser.parse_args()

    fp_views: str = "data/views_data/"  # path to views data
    clusters: str = "hdbscan"  # clustering method
    prediction_windows: list[int] = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    level: Literal["cm", "pgm"] = "pgm"

    ### use this to test changes
    # If true, this will overwrite some config parameters before starting the run
    test_mode: bool = args.test

    # these wrappers handle custom kwargs/options for predictors to get consistent output
    prediction_wrappers: dict[str, Callable] = {
        "DRF": drf_predict,
        "QRF": qrf_predict,
        "NGB": ngb_predict,
    }

    # we use precision recall by default for classifier tuning
    scorer_clf: dict[str, Callable] = {
        "RF": make_scorer(average_precision_score, response_method="predict_proba"),
        "XGB": make_scorer(average_precision_score, response_method="predict_proba"),
    }

    # freeze the estimator-specific prediction function in the general scorer function
    scorer_reg: dict[str, Callable] = {
        key: partial(crps_scorer, func=item) for key, item in prediction_wrappers.items()
    }

    # implementation is based on this structure (model_type: {space: {}, max_evals: int})
    tuning_space_clf: dict[str, dict[str, Any]] = {
        "RF": {
            "space": {
                "max_depth": hp.uniformint(label="max_depth", low=3, high=8),
                "n_estimators": hp.quniform(label="n_estimators", low=40, high=1000, q=20),
            },
            "max_evals": 30,
        },
        "XGB": {
            "space": {
                "max_depth": hp.uniformint(label="max_depth", low=3, high=20, q=1),
                "n_estimators": hp.uniformint(label="n_estimators", low=40, high=1001, q=50),
                "learning_rate": hp.choice(
                    label="learning_rate", options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3]
                ),  # important
                "gamma": hp.uniform(label="gamma", low=1, high=9),  # less important
                "subsample": hp.quniform(label="subsample", low=0.5, high=1.0, q=0.1),
                "colsample_bytree": hp.quniform(
                    label="colsample_bytree", low=0.5, high=1, q=0.1
                ),  # less important
                "reg_alpha": hp.uniformint(
                    label="reg_alpha", low=0, high=100, q=1
                ),  # less important
                "reg_lambda": hp.uniform(label="reg_lambda", low=0, high=1),  # less important
                "min_child_weight": hp.loguniform(
                    label="min_child_weight", low=-3, high=2.75
                ),  # important
                "max_delta_step": hp.uniform(label="max_delta_step", low=1, high=10),
            },
            "max_evals": 100,
        },
    }

    tuning_space_reg: dict[str, dict[str, Any]] = {
        "DRF": {
            "space": {
                "num_features": hp.uniformint(label="num_features", low=5, high=50, q=1),
                "num_trees": hp.quniform(label="num_trees", low=40, high=1000, q=20),
                "min_node_size": hp.uniformint(label="min_node_size", low=5, high=20, q=1),
                "alpha": hp.quniform(label="alpha", low=0.01, high=0.2, q=0.01),  # less important
            },
            "max_evals": 50,
        },
        "QRF": {
            "space": {
                "n_estimators": hp.quniform(
                    label="n_estimators", low=160, high=1000, q=20
                ),  # fairly important
                "max_depth": hp.uniformint(label="max_depth", low=3, high=6),  # important
                "criterion": hp.choice(
                    label="criterion", options=["absolute_error"]
                ),  # apparently not as good: ['squared_error', 'friedman_mse', 'poisson']
                "min_samples_split": hp.uniformint(
                    label="min_samples_split", low=5, high=20, q=1
                ),  # less important
                "min_samples_leaf": hp.uniformint(label="min_samples_leaf", low=10, high=40, q=1),
                "max_features": hp.choice(
                    label="max_features", options=["log2"]
                ),  # ['sqrt', None] seem worse
            },
            "max_evals": 50,
        },
        "NGB": {
            "space": {
                "n_estimators": hp.quniform(label="n_estimators", low=200, high=500, q=20),
                "learning_rate": hp.choice(label="learning_rate", options=[0.001, 0.005, 0.01]),
                "base_criterion": hp.choice(
                    label="base_criterion", options=["squared_error", "friedman_mse"]
                ),  # not important, both better than 'absolute_error'
                "base_splitter": hp.choice(
                    label="base_splitter", options=["random"]
                ),  # 'random' apparently better than 'best'
                "base_max_depth": hp.uniformint(label="base_max_depth", low=2, high=4),
                "minibatch_frac": hp.quniform(label="minibatch_frac", low=0.1, high=1, q=0.1),
                "col_sample": hp.quniform(label="col_sample", low=0.1, high=1, q=0.1),
            },
            "max_evals": 50,
        },
    }

    hurdle_kwargs = {
        "scorer_clf": scorer_clf,
        "scorer_reg": scorer_reg,
        "tuning_space_clf": tuning_space_clf,
        "tuning_space_reg": tuning_space_reg,
        "prediction_wrappers": prediction_wrappers,
    }

    # kwargs all have default values
    prediction_kwargs: dict[str, Any] = {
        "regenerate_predictions": False,
        "ignore_prior_tuning": False,
        "years_prior": 3,
        # 'cluster_kwargs': {
        #     'visualize_clusters': False,
        #     'regenerate_clusters': False,
        # },
        "random_state": 1,
        "calibration_kwargs": {
            "calibrate_classifier": False,
            "calibration_months": 36,
            # 'min_positives': 200
        },
    }

    if level == "cm":
        clusters = "cm"

    # update config if test_mode - since we're using globals for the config it had to be done here
    if test_mode:
        # use test filepaths to not accidentally overwrite anything
        clusters = f"{level}_test"
        # reduce tuning trials to 2 via dict comprehension in tuning spaces
        hurdle_kwargs["tuning_space_reg"] = {
            key_outer: {
                key_inner: (item_inner if key_inner != "max_evals" else 2)
                for key_inner, item_inner in item_outer.items()
            }
            for key_outer, item_outer in tuning_space_reg.items()
        }
        hurdle_kwargs["tuning_space_clf"] = {
            key_outer: {
                key_inner: (item_inner if key_inner != "max_evals" else 2)
                for key_inner, item_inner in item_outer.items()
            }
            for key_outer, item_outer in tuning_space_clf.items()
        }
        # calibration does not seem to work with current test data setup
        prediction_kwargs["calibration_kwargs"] = {
            key: False if key == "calibrate_classifier" else item
            for key, item in prediction_kwargs["calibration_kwargs"].items()
        }

    # run pipeline
    now = time.time()

    for year in prediction_windows:
        ensemble_predictions(year)
    print("Done in", time.time() - now, "seconds.")
