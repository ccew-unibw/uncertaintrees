"""
Collection of functions handling the hyperparameter tuning via hyperopt.
"""

import os
import pickle
from collections.abc import Generator, Callable
from typing import Any

from hyperopt import STATUS_OK, Trials, fmin, tpe, space_eval
import numpy as np
import pandas as pd
import pendulum
from quantile_forest import RandomForestQuantileRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src.estimators.drf import DistributionalRandomForestRegressor
from src.estimators.ngboost import NGBRegressorWrapper
from src.utils.conversion import get_date, get_month_id


def make_cv_index(df: pd.DataFrame, single_fold: bool = False) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """
    Cross validation index generator based on a 5-year moving window bounded at the beginning
    and end of calendar years. Each resulting fold consists of 3 years training data, 1 year gap
    and 1 year test data and replicates the data cutoff in October.
    
    Args:
        df (pd.DataFrame): Data for which to generate the CV index
        single_fold (bool, optional): Flag whether to skip CV splits generation and simply return
            train/test split. Necessary for consistency with the modelling pipeline when CV is not 
            possible due to small shares of non-zero target.

    Returns:
        Generator[tuple[np.ndarray, np.ndarray]]: CV generator with tuples of train, test int
        indices
    """
    # make sure its the right format
    assert df.index.names[0] == "month_id" and df.index.names[1] in [
        "priogrid_gid",
        "country_id",
    ]
    df = df.copy()
    # if we have a series
    if type(df) is pd.Series:
        df = df.to_frame()
    df["int_index"] = [i for i in range(len(df))]
    min_date = get_date(df.index.get_level_values("month_id").min())
    max_date = get_date(df.index.get_level_values("month_id").max())
    if not single_fold:
        try:
            assert max_date.subtract(years=10) > min_date
            # to include the last year below, we need to go to the beginning of the next year
            period = pendulum.period(
                min_date.add(years=5), max_date.end_of("year").add(microseconds=1)
            )
            for date in period.range("years"):
                train_start = get_month_id(date.subtract(years=5))
                train_end = get_month_id(date.subtract(years=2)) - 3
                train_index = df.loc[train_start:train_end].int_index.to_numpy()
                test_start = get_month_id(date.subtract(years=1))
                test_end = (
                    get_month_id(date) - 3
                )  # always set to end of year - doesnt matter for indexing if this is after the last month id
                test_index = df.loc[test_start:test_end].int_index.to_numpy()
                yield (train_index, test_index)
        except AssertionError:  # we want at least
            raise ValueError("Please provide at least 10 years of data with passed df.")
    else:
        train_start = get_month_id(min_date)
        train_end = (
            get_month_id(get_date(df.iloc[round(0.70 * len(df))].name[0]).end_of("year")) - 2 # type: ignore
        )
        train_index = df.loc[train_start:train_end].int_index.to_numpy()
        test_start = train_end + 15
        test_end = get_month_id(max_date)
        test_index = df.loc[test_start:test_end].int_index.to_numpy()
        yield (train_index, test_index)


def model_selection(model_type: str, random_state: int, params: dict[str, Any] | None = None):
    """
    Base model selection based on model_type string with random state.
    Optionally also sets model params if passed.
    
    Args:
        model_type (str): internal Model string ID
        random_state (int): random state passed to models
        params (dict[str, Any] | None, optional): Estimator params, set if provided.

    Returns:
        Initialized estimator instance.
    """
    available_models = {
        "XGB": XGBClassifier(random_state=random_state, n_jobs=-1),
        "RF": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "logit": LogisticRegression(random_state=random_state, n_jobs=-1),
        "DRF": DistributionalRandomForestRegressor(
            random_state=random_state
        ),  # default is to use as many jobs as cores
        "QRF": RandomForestQuantileRegressor(random_state=random_state, n_jobs=-1),
        "NGB": NGBRegressorWrapper(random_state=random_state),  # no n_jobs available
    }

    try:
        model = available_models[model_type]
    except KeyError:
        raise ValueError(
            f'Model_type parameter "{model_type}" invalid. '
            f'Needs to be one of ["RF" , "XGB", "logit", "DRF", "QRF", "NGB"].'
        )

    # add params to models - space is essentially just like a parameter dictionary for the model
    if params is not None:
        model.set_params(**params)

    return model


def hyper_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    space: dict,
    scorer: Callable,
    fp: str,
    random_state: int,
    penalty_weight: float = 0.5,
    max_evals: int = 20,
    do_cv: bool = True,
) -> tuple[dict, float]:
    """
    Function performs hyperopt hyperparameter tuning and stores all results in a pickled Trials
    object.

    Args:
        X (pd.DataFrame): dataframe with features.
        y (pd.Series): series with target.
        model_type (str): model type key used to select and build the base model.
        space (dict): hyperopt tuning space for the model.
        scorer (Callable): sklearn compatible scorer used in cross-validation.
        fp (str): filepath to store tuning results (dict[hyperopt Trials object, space])
        random_state (int): random state passed to model.
        penalty_weight (float): weight of the penalty for deviation between mean train and test 
            scores for the final "loss".
        max_evals (int, optional): number of tuning trials to run.
        do_cv (bool, optional): whether to use cross-validation or not.

    Returns:
        tuple[dict, float]: dictionary with best model parameters, best loss
    """

    def objective(params):
        """
        Function producing the score we want to minimize. We use the mean test score from
        cross-validation penalized by deviations between mean test and mean training scores to
        emphasize generalizability and guard against overfitting.
        """
        model = model_selection(model_type, random_state, params)
        # hotfix for the "float" problem with hyperopt hp.quniform space definition
        if "n_estimators" in model.get_params():
            model.set_params(n_estimators=int(model.n_estimators))

        if do_cv:
            cv = make_cv_index(X)
        else:
            cv = make_cv_index(X, single_fold=True)
        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y,
            scoring=scorer,
            cv=cv,
            return_train_score=True,
            n_jobs=-1,
        )
        test_score = np.nanmean(cv_results["test_score"])
        train_score = np.nanmean(cv_results["train_score"])
        penalty = abs(train_score - test_score)  # penalize deviations between train and test scores
        penalized_score = (
            test_score - penalty_weight * penalty
        )  # weight could be changed by model type (or even as a hyperparameter)
        loss = -penalized_score  # loss is negative as it needs to be minimized
        # print('test:', test_score, 'train:', train_score, '"loss":', loss)

        return {
            "loss": loss,
            "status": STATUS_OK,
            "attachments": {
                "mean_test_score": test_score,
                "mean_train_score": train_score,
            },
        }

    trials = Trials()  # this stores information on individual runs

    _ = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(4651465763132369873614154318718465),
    )

    best_hyperparams = space_eval(space, trials.argmin)

    with open(fp, "wb") as f:
        pickle.dump({"trials": trials, "space": space}, f)

    # returns best params and best loss for comparison with other models
    return best_hyperparams, min(trials.losses())


def build_model(model_type: str, param_dict: dict[str, Any], random_state: int):
    """
    Function to build the corresponding model after the hyperparameter tuning above was performed.
    
    Args:
        model_type (str): internal Model string ID
        param_dict (dict[str, Any]): Estimator params
        random_state (int): random state passed to models
        
    Returns:
        initialized Estimator
    """
    # get base model
    model = model_selection(model_type, random_state)

    # where possible, do int conversion from float - required for e.g. RF
    for param in param_dict:
        if type(param_dict[param]) is not str and param_dict[param] == int(param_dict[param]):
            param_dict[param] = int(param_dict[param])
    model = model.set_params(**param_dict)

    return model


def load_tuning_results(fp: str) -> tuple[dict | None, float]:
    """
    Load stored trials and space and return best params and corresponding loss.
    
    Args:
        fp (str): filepath to stored tuning results
    
    Returns:
        tuple[dict | None, float]: best model params, best loss
    """
    with open(fp, "rb") as f:
        trials_dict = pickle.load(f)

    if trials_dict == "unable to tune":
        return {}, np.inf
    trials = trials_dict["trials"]
    space = trials_dict["space"]

    # if file exists but is only a string, we were unable to tune the local model but don't want the pipeline to fail
    if type(trials) is str:
        print(trials)
        return None, float("inf")

    best_params = space_eval(space, trials.argmin)
    best_loss = min(trials.losses())

    return best_params, best_loss


# additional functions for manual checks - not used in the pipeline
def get_trials(
    model_type: str,
    target: str,
    fp_tuning: str,
    local_model: bool = False,
    cluster: str | int = "",
) -> tuple[Trials, dict]:
    """load stored Trials object and search space"""
    folder = target[target.rfind("_") + 1 :]
    if local_model:
        sub_folder = os.path.join("local", f"c{cluster}")
    else:
        sub_folder = "global"
    fp = os.path.join(fp_tuning, folder, sub_folder)
    filename = f"{model_type}{f'_local{cluster}' if local_model else '_global'}_{target[target.rfind('_') + 1 :]}_trial_results.pkl"
    with open(os.path.join(fp, filename), "rb") as f:
        trials_dict = pickle.load(f)
    trials = trials_dict["trials"]
    space = trials_dict["space"]
    return trials, space


def get_trial_params(trial, space):
    trial_params = {}
    for key in trial["misc"]["vals"]:
        trial_params[key] = trial["misc"]["vals"][key][0]

    best_trial = space_eval(space, trial_params)
    return best_trial


def create_tuning_results_df(
    model_type: str,
    target: str,
    fp_tuning: str,
    local_model: bool = False,
    cluster: str | int = "",
) -> pd.DataFrame:
    """
    Creates a dataframe for manual comparison of performance across different trials / the
    parameter space. Used for manual checks and refining the hyperparameter search space, not
    directly in the modeling pipeline.
    """
    trials, space = get_trials(
        model_type, target, fp_tuning, local_model=local_model, cluster=cluster
    )
    columns = ["loss", "mean_train_score", "mean_test_score"] + list(space.keys())
    df_results = pd.DataFrame(columns=columns)
    for trial in trials.trials:
        scores = {
            "loss": trial["result"]["loss"],
            "mean_train_score": trials.trial_attachments(trial)["mean_train_score"],
            "mean_test_score": trials.trial_attachments(trial)["mean_test_score"],
        }
        trial_params = get_trial_params(trial, space)

        df_results = pd.concat([df_results, pd.DataFrame([{**scores, **trial_params}])])
    df_results = df_results.sort_values("loss").reset_index(drop=True)
    return df_results
