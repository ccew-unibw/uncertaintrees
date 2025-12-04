"""
Custom hurdle "estimator" designed to automate tuning and model selection for this specific
prediction problem. Used to generate "raw" predictions, which are then combined through the
GlobalLocalHurdleEnsemble.

NOTE: unlike some of the other estimators here not designed to be sklearn compatible...
would have required a different approach with regard to tuning etc. and more
"""

import os
import pickle
from typing import Any
import warnings
from collections.abc import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from src.utils.model_tuning import hyper_optimization, build_model, load_tuning_results


class HurdleStepsWithTuning(BaseEstimator):
    """
    This class is designed to automate the hyperparameter tuning and selection of the
    best-performing model for a number of possible classifiers and regressors as part of a hurdle
    ensemble. The tuning is performed with hyperopt spaces and sklearn cross_validate, so
    estimators need to be compatible with sklearn.

    It can further be used to jointly run fit and predict for the different components of the
    hurdle model.

    NOTE: Generally skipping input data validation in this class directly, since it is expected to
    be performed by the estimators anyways and allows the class to stay more flexible.

    Currently implemented estimators in the project:
        Classifiers:
            "RF": Random Forest Classifier
            "XGB": XGBoost Classifier
        Regressors
            "DRF": Distributional Random Forest Regressor
            "QRF": Quantile Random Forest Regressor
            "NGB": NGBoost Regressor

    """
    def __init__(
        self,
        scorer_clf: dict[str, Callable],
        scorer_reg: dict[str, Callable],
        tuning_space_clf: dict[str, dict],
        tuning_space_reg: dict[str, dict],
        prediction_wrappers: dict[str, Callable],
        fp_tuning: str = "src/tuning_trials",
        random_state: int = 1,
    ):
        """Initializes Class

        Args:
            scorer_clf (dict[str, Callable]): dictionary with estimator keys and corresponding
                sklearn compatible scorers for classifiers
            scorer_reg (dict[str, Callable]): dictionary with estimator keys and corresponding
                sklearn compatible scorers for regressors
                NOTE: for the model selection to work as expected, the scorers need to implement
                the same score within one step of the hurdle model!
            tuning_space_clf (dict[str, dict]): dictionary defining the available classification
                estimators and corresponding tuning spaces.
                Required format:
                    tuning_space_clf = {
                        '<estimator_key1>': {
                            'space': {<hyperopt tuning space>},
                            'max_evals': <number of tuning trials to run (int)>
                        },
                        '<estimator_key2>': {
                            'space': {<hyperopt tuning space>},
                            'max_evals': <(int)>
                        },
                        ...
                    }
            tuning_space_reg (dict[str, dict]): dictionary defining the available regression
                estimators and corresponding tuning spaces. See above for format.
            prediction_wrappers (dict[str, Callable]): (optional) dictionary with estimator keys
                and wrapper functions to be called during prediction with func(estimator, X).
                Encompasses both classifiers and regressors. If key is not present, an estimators
                standard prediction method .predict(X)/.predict_proba(X) will be called.
            fp_tuning (str): filepath to store tuning results
            random_state (int): random state to be passed to estimators
        """
        self.fp_tuning = fp_tuning
        self.random_state = random_state

        # tuning space and scorer dicts need to fit together, i.e. have the same keys
        try:
            assert sorted(tuning_space_clf.keys()) == sorted(scorer_clf.keys())
            self.scorer_clf = scorer_clf
            self.tuning_space_clf = tuning_space_clf
        except AssertionError:
            raise ValueError(
                f"Tuning space and scorer need to fit together, i.e. have the same keys. For "
                f"classification got {tuning_space_clf.keys()} and {scorer_clf.keys()}."
            )

        try:
            assert sorted(tuning_space_reg.keys()) == sorted(scorer_reg.keys())
            self.scorer_reg = scorer_reg
            self.tuning_space_reg = tuning_space_reg
        except AssertionError:
            raise ValueError(
                f"Tuning space and scorer need to fit together, i.e. have the same keys. For regression "
                f"got {tuning_space_reg.keys()} and {scorer_reg.keys()}."
            )

        try:
            assert sorted({**scorer_clf, **scorer_reg}.keys()) == sorted(prediction_wrappers.keys())
            self.prediction_wrappers = prediction_wrappers
        except AssertionError:
            prediction_wrappers_filled = dict.fromkeys({**scorer_clf, **scorer_reg}.keys())
            for model_type in prediction_wrappers_filled:  # Fill incomplete dictionaries
                try:
                    prediction_wrappers_filled[model_type] = prediction_wrappers[model_type]
                except KeyError:
                    prediction_wrappers_filled[model_type] = (
                        None  # in this case no special wrapper will be used
                    )
                    warnings.warn(
                        f"No wrapper defined for model type {model_type}. Using default predict method."
                    )
            self.prediction_wrappers = prediction_wrappers_filled

    def get_tuning_fp_(self, target: str, local_model: bool, cluster: int | str) -> str:
        """helper method to create the filepath of the storage folder"""
        folder = target[target.rfind("_") + 1 :]
        if local_model:
            sub_folder = os.path.join("local", f"c{cluster}")
        else:
            sub_folder = "global"
        fp = os.path.join(self.fp_tuning, folder, sub_folder)
        return fp

    def generate_tuning_results_filename_(
        self, model_type: str, target: str, local_model: bool, cluster: int | str
    ) -> str:
        """helper method to create the filename of the tuning results file"""
        filename = (
            f"{model_type}{f'_local{cluster}' if local_model else '_global'}_{target[target.rfind('_') + 1 :]}"
            f"_trial_results.pkl"
        )
        return filename

    def make_hurdle_datasets_(
        self, df_train: pd.DataFrame, target: str, features: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """helper method to create the different views of the training data for each component of the hurdle model"""
        target_reg = target
        target_clf = "dummy_" + target
        X_clf = df_train[features]
        y_clf = df_train[target_clf]

        X_reg = df_train[df_train[target_reg] > 0][features]
        y_reg = df_train[df_train[target_reg] > 0][target_reg]
        return X_clf, X_reg, y_reg, y_clf

    def tune(
        self,
        df_train: pd.DataFrame,
        features: list[str],
        target: str,
        local_model: bool = False,
        cluster: int | str = "",
        ignore_prior_tuning: bool = False,
    ):
        """
        Tunes the available classifiers and regressors as specified in the tuning_space_
        variables and selects the best regressor respectively classifier. This method simply
        splits the training data and passed the respective data to the tune_models_ method where
        the actual tuning logic is handled.

        Args:
            df_train (pd.DataFrame): dataframe with training data (features and regression &
                classification targets combined)
            features (list[str]): list of features for training
            target (str): name of (regression) target column (dummy version for classifier
                selected automatically)
            local_model (bool): whether it is a local or global model in our overall ensemble -
                important mainly for the storage logic (filepath/name) and because a failure to
                tune a local model (e.g. due to lack of data) will result in the params of the
                global model for that target to be selected if available.
            cluster (int | str): cluster id
            ignore_prior_tuning (bool): whether to reload existing tuning results where
                available (useful e.g. if new estimators are implemented)

        """
        X, X_reg, y_reg, y_clf = self.make_hurdle_datasets_(df_train, target, features)

        print("tune classifier...")
        clf, clf_model_type = self.tune_models_(X, y_clf, local_model, cluster, ignore_prior_tuning)
        self.clf = clf
        self.clf_model_type = clf_model_type
        print("tune regressor...")
        reg, reg_model_type = self.tune_models_(
            X_reg, y_reg, local_model, cluster, ignore_prior_tuning
        )
        self.reg = reg
        self.reg_model_type = reg_model_type
        print("...done")
        self.is_tuned_ = True

    def tune_models_(self, X, y, local_model, cluster, ignore_prior_tuning) -> tuple[Any, str]:
        """
        'Private' method to handle the actual tuning logic:
        - Iterates through available model types
        - Calls the function handling the actual tuning with cross_validate and hyperopt
        - selects the best-performing model and its config by storing the result only if
          better than previous best
        - builds and returns the best-performing model and corresponding model_type
        Best-performing estimators and corresponding model keys (types) are stored as class
        variables.
        """
        target = y.name
        # get folder for output
        fp = self.get_tuning_fp_(target, local_model, cluster)
        if not os.path.exists(fp):
            os.makedirs(fp)

        # select tuning space based on passed y
        if "dummy" in target:
            tuning_space = self.tuning_space_clf
            scorers = self.scorer_clf
        else:
            tuning_space = self.tuning_space_reg
            scorers = self.scorer_reg

        # perform hyperparamter tuning - results are also saved to disk
        best_model_type = None
        best_model_params = None
        best_loss = float("inf")  # intialize loss with high value - hyperopt is trying to minimize

        # tune for each model type specified in space and identify model with best loss for this target
        for model_type in tuning_space:
            filename = self.generate_tuning_results_filename_(
                model_type, target, local_model, cluster
            )
            trials_fp = os.path.join(fp, filename)
            if os.path.exists(trials_fp) and not ignore_prior_tuning:
                print(f"{model_type} model already tuned...")
                best_params, loss = load_tuning_results(trials_fp)
                # print('tuned model params, loss', best_params, loss)
            else:
                space = tuning_space[model_type]["space"]
                scorer = scorers[model_type]
                max_evals = tuning_space[model_type]["max_evals"]
                # run optimization
                try:
                    best_params, loss = hyper_optimization(
                        X,
                        y,
                        model_type,
                        space,
                        scorer,
                        fp=trials_fp,
                        random_state=self.random_state,
                        max_evals=max_evals,
                    )
                except Exception as e:
                    print(e)
                    try:
                        best_params, loss = hyper_optimization(
                            X,
                            y,
                            model_type,
                            space,
                            scorer,
                            fp=trials_fp,
                            random_state=self.random_state,
                            max_evals=max_evals,
                            do_cv=False,
                        )
                        print(f"CV skipped for {model_type} model")
                    except:
                        print(f"Couldn't tune local {model_type} model.")
                        with open(trials_fp, "wb") as f:
                            pickle.dump("unable to tune", f)
                        continue
            if loss < best_loss:
                best_model_type = model_type
                best_model_params = best_params
                best_loss = loss

        # get global/default model & params if local cannot be tuned
        if best_model_type is None and local_model:
            print(
                "No local model could be tuned. Defaulting to global params of default estimator."
            )
            # use first model type in tuning space as default
            best_model_type = list(tuning_space.keys())[0]
            try:
                # the fp functions default to global
                fp_global = self.get_tuning_fp_(target, local_model, cluster)
                filename_global = self.generate_tuning_results_filename_(
                    best_model_type, target, local_model, cluster
                )
                global_trials_fp = os.path.join(fp_global, filename_global)
                best_model_params = load_tuning_results(global_trials_fp)[0]
            except Exception as e:
                print(e)
                print("No global model found. Using default model params")
                best_model_params = {}
        # create best model
        tuned_model = build_model(best_model_type, best_model_params, self.random_state) # type: ignore
        print("best model type:", best_model_type)
        print("best model params:", best_model_params)
        return tuned_model, best_model_type # type: ignore

    def fit(self, df_train: pd.DataFrame, features: list[str], target: str):
        """
        Selects relevant training data views and calls fit on the regressor/classifier with the
        best performance during tuning.
        """
        try:
            assert self.is_tuned_
        except AssertionError:
            print("No tuned models available, yet! Please call estimator.tune() first.")
            raise

        X, X_reg, y_reg, y_clf = self.make_hurdle_datasets_(df_train, target, features)
        self.clf.fit(X, y_clf)
        self.reg.fit(X_reg, y_reg)
        self.is_fitted_ = True

    def predict(self, X, y=None) -> tuple[Any, Any]:
        """
        Simple prediction method which calls either the standard predict(X)/predict_proba(X)
        method of the estimator or the wrapper function as defined in the class variable
        prediction_wrappers.

        Returns the separate predictions "raw" to allow processing and combining the predictions
        in various ways outside this class.
        """
        check_is_fitted(self, "is_fitted_")
        clf_wrapper = self.prediction_wrappers[self.clf_model_type]
        reg_wrapper = self.prediction_wrappers[self.reg_model_type]
        if clf_wrapper is not None:
            preds_clf = clf_wrapper(
                self.clf, X
            )  # no kwargs functionality for classifiers implemented
        else:
            try:
                preds_clf = self.clf.predict_proba(X)
            except Exception as e:
                print("Error:", e)
                raise ValueError(
                    "Default predict_proba(X) method failed, consider defining a wrapper function for the "
                    "classifier."
                )
        if reg_wrapper is not None:
            preds_reg = reg_wrapper(self.reg, X)
        else:
            try:
                preds_reg = self.reg.predict(X)
            except Exception as e:
                print("Error:", e)
                raise ValueError(
                    '"Default" predict(X) method failed, consider defining a wrapper function for the '
                    "regressor."
                )
        return preds_clf, preds_reg
