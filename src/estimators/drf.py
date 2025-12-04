"""
Python wrapper for Distributional Random Forests R implementation
(https://github.com/lorismichel/drf). Modified version of the author's wrapper for
(limited) compatibility with sklearn.

Usage example:
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # generate data - simple linear regression
    X, y = make_regression(n_samples=10000, n_features=1, noise=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # fit model
    DRF = DistributionalRandomForestRegressor()
    DRF.fit(X_train, y_train)
    # mean gives point predictions, sample samples based on the weights (which is what we need
    # for the competition)
    # out is a dictionary in the case of multiple functionals, else a numpy array
    out = DRF.predict(X_test, functional=['mean', 'sample'])

    # visualize point predictions - blue is predictions, red real values
    fig, ax = plt.subplots(1,1)
    ax.scatter(X_test, y_test, s=.2, color='blue')
    ax.scatter(X_test, out['mean'], s=.2, color='red')
    plt.show()

"""

import warnings
from collections.abc import Callable
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri, packages
from rpy2.robjects.vectors import StrVector

# suppress R warnings
rpy2_logger.setLevel(logging.ERROR)

numpy2ri.activate()
pandas2ri.activate()

# try-except to run the install only if the import fails
try:
    base_r_package = packages.importr("base")
    drf_r_package = packages.importr("drf")
except:
    # had to add the package installation here based on the rpy2 documentation
    # this should hopefully only install if not already installed - base r installs a whole boatload of stuff

    # import rpy2's package module
    # import R's utility package
    utils = packages.importr("utils")
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
    # R package names
    packnames = ("base", "drf")

    # Selectively install what needs to be installed.
    # We are fancy, just because we can.
    names_to_install = [x for x in packnames if not packages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    # lets try the import again
    base_r_package = packages.importr("base")
    drf_r_package = packages.importr("drf")


def w_cov(x, y, w):
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    return np.average((x - mx) * (y - my), weights=w)


def w_quantile(values, quantiles, sample_weight=None, values_sorted=False):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


class DistributionalRandomForestRegressor(BaseEstimator):
    """
    Sklearn-compatible wrapper for R implementation of Distributional Random Forests
    (https://jmlr.org/papers/v23/21-0585.html), adapted from https://github.com/lorismichel/drf.

    See the R documentation (https://cran.r-project.org/web/packages/drf/drf.pdf) for more
    information on the hyperparameters and functionals.
    """

    # not using the RegressorMixin since the Multioutput functionality and the different output
    # types of calling predict make it hard to define a universal score function
    _estimator_type = "regressor"

    # added all parameters from R package here individually, so we can hyperparameter tune in sklearn
    def __init__(
        self,
        num_trees=500,
        splitting_rule="FourierMMD",
        num_features=10,
        bandwidth=None,
        response_scaling=False,
        node_scaling=False,
        sample_weights=None,
        sample_fraction=0.5,
        mtry=None,
        min_node_size=15,
        honesty=False,
        honesty_fraction=0.5,
        honesty_prune_leaves=None,
        alpha=0.05,
        imbalance_penalty=0,
        compute_oob_predictions=False,
        num_threads=None,
        compute_variable_importance=False,
        # this is called "seed" in the R function and replaced during fit
        random_state=None,
    ):  # ok
        self.num_trees = num_trees
        self.splitting_rule = splitting_rule
        self.num_features = num_features
        self.bandwidth = bandwidth
        self.response_scaling = response_scaling
        self.node_scaling = node_scaling
        self.sample_weights = sample_weights
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.compute_oob_predictions = compute_oob_predictions
        self.num_threads = num_threads
        self.compute_variable_importance = compute_variable_importance
        self.random_state = (
            random_state  # replacing "seed" with "random_state" here to be sklearn compatible
        )

    def _more_tags(self):
        return {"multioutput": True, "requires_y": True}

    def fit(self, X, y):
        # checks
        X, y = super()._validate_data(X, y, multi_output=True)
        if len(y.shape) == 1:  # save target dims as attribut
            self.y_dims_ = 1
        else:
            self.y_dims_ = y.shape[1]

        self.X_ = X
        self.y_ = y

        # I don't quite get why but this apparently needs to be a dataframe
        # before conversion to R for the drf to work
        X_r = ro.conversion.py2rpy(pd.DataFrame(X))
        y_r = ro.conversion.py2rpy(pd.DataFrame(y))

        # use the get_params() functionality so we can easily pass them on (replace "random_state" with "seed" for R)
        params_r = {
            "seed" if key == "random_state" else key: value
            for key, value in super().get_params().items()
            if value is not None
        }

        self.r_fit_object = drf_r_package.drf(X_r, y_r, **params_r)
        self.is_fitted_ = True
        return self

    def info(self):  # ok
        drf_r_package.print_drf(self.r_fit_object)

    def variable_importance(self):  # ok
        ro.r('sink("/dev/null")')
        if self.r_fit_object.variable_importance is None:
            ret = drf_r_package.variableImportance(self.r_fit_object)
        else:
            ret = self.r_fit_object.variable_importance
        ro.r("sink()")
        return ret

    def predict(
        self,
        X,
        functional: str | list[str] = "weights",
        transformation: Callable | None = None,
        **predict_kwargs,
    ):
        """
        Default return are simply the weights, keeping with the R implementation.

        Args:
            X: array-like of shape(N, self.n_features_in_)
                New data to predict unknown ys from
            functional (str | list[str], optional):
                Multiple statistical functionals can be calculated by specifying the
                "functional" parameter:
                - "weights" : the underlying weights for the functionals calculated from X (default)
                - "mean" : conditional mean
                - "sd" : conditional standard deviation
                - "cov" : conditional covariance
                - "cor" : conditional correlation
                - "quantile" : conditional quantiles
                - "sample" : conditional weighted sample
            transformation (Callable | None, optional): Transformation to be applied to the
                training y before using it to calculate statistical functionals (not fully
                tested). If no function is supplied, no transformation is performed.
            **predict_kwargs:
                Two further kwargs are required to calculate specific functionals:
                "quantiles" : array-like[float],
                    quantiles to calculate for "quantile" functional
                "n_samples" : int
                    number of samples for "sample" functional

        Returns:
            A single array if type(functional) is str, else a dictionary with the "functional"
            list as keys and the output arrays as values.

        """

        def get_weights(weights, y, **kwargs):
            """
            simply the weights calculated for X_test
            Returns array with shape (len(X_test), len(y_train))
            """
            return weights

        def get_mean(weights, y, **kwargs):
            """
            conditional mean - mean prediction based on weights calculated from test X and
            multiplied with the training ys
            Returns array with shape (len(X_test), y dims)
            """
            mean = np.zeros((X.shape[0], y.shape[1]))
            y = pd.DataFrame(y)

            for i in range(X.shape[0]):
                mean[i, :] = y.multiply(weights[i, :], axis=0).sum().to_numpy()
            return mean

        def get_sd(weights, y, **kwargs):
            """
            conditional standard deviation for predictions
            Returns array with shape (len(X_test), y dims)
            """
            sd = np.zeros((X.shape[0], y.shape[1]))

            for i in range(X.shape[0]):
                for j in range(y.shape[1]):
                    sd[i, j] = w_cov(y[:, j], y[:, j], weights[i, :]) ** 0.5
            return sd

        def get_cov(weights, y, **kwargs):
            """
            conditional covariance
            Returns array with shape (X_test.shape[0], y_train.shape[1], y_train.shape[1])
            """
            cov = np.zeros((X.shape[0], y.shape[1], y.shape[1]))

            for i in range(X.shape[0]):
                for j in range(y.shape[1]):
                    for k in range(y.shape[1]):
                        cov[i, j, k] = w_cov(y[:, j], y[:, k], weights[i, :])
            return cov

        def get_cor(weights, y, **kwargs):
            """
            conditional correlation
            Returns array with shape (X_test.shape[0], y_train.shape[1], y_train.shape[1])
            """
            cor = np.zeros((X.shape[0], y.shape[1], y.shape[1]))

            for i in range(X.shape[0]):
                for j in range(y.shape[1]):
                    for k in range(y.shape[1]):
                        cov = w_cov(y[:, j], y[:, k], weights[i, :])
                        sd1 = w_cov(y[:, j], y[:, j], weights[i, :]) ** 0.5
                        sd2 = w_cov(y[:, k], y[:, k], weights[i, :]) ** 0.5
                        cor[i, j, k] = cov / (sd1 * sd2)

            return cor

        def get_quantile(weights, y, **kwargs):
            """
            conditional quantiles of predictions
            Returns array with shape (X_test.shape[0], y_train.shape[1], len(quantiles))
            """
            if "quantiles" in kwargs.keys():
                quantile_list = kwargs["quantiles"]
                assert type(quantile_list) is not str
            else:
                warnings.warn(
                    "Quantiles not specified, using default quantiles [0.1, 0.5, 0.9]. "
                    + 'To change this, specify quantiles via argument "quantiles" as array-like '
                    + "in your call of drf.predict().",
                    Warning,
                )
                quantile_list = [0.1, 0.5, 0.9]
            # print(quantiles)
            quantile = np.zeros((X.shape[0], y.shape[1], len(quantile_list)))

            for i in range(X.shape[0]):
                for j in range(y.shape[1]):
                    quantile[i, j, :] = w_quantile(
                        y[:, j], quantile_list, sample_weight=weights[i, :]
                    )

            return quantile

        def get_sample(weights, y, **kwargs):
            """
            conditional quantiles of predictions
            Returns array with shape (X_test.shape[0], y_train.shape[1], n_samples)
            """
            if "n_samples" in kwargs.keys():
                n = kwargs["n_samples"]
                if type(n) is not int:
                    try:
                        n = int(n)
                    except ValueError:
                        raise ValueError(
                            "n_samples needs to be of type int or coercible into type int"
                        )
            else:
                warnings.warn(
                    "No sample size specified, using default sample size n=100. To change this, specify "
                    + 'sample size via argument "n_samples" in your call of drf.predict().',
                    UserWarning,
                )
                n = 100
            # modified from source for speed reasons, but quite confident its the same
            sample = np.zeros((X.shape[0], y.shape[1], n))
            rng = np.random.default_rng(290421007664171484893943332734122392547)
            idx_array = np.array(range(y.shape[0]))
            for i in range(X.shape[0]):
                p = weights[i, :]
                ids = rng.choice(idx_array, n, p=p)
                sample[i, :, :] = y[ids, :].T

            return sample

        # checks
        X = super()._validate_data(X, reset=False)
        check_is_fitted(self, "is_fitted_")
        X = pd.DataFrame(X)
        X_r = ro.conversion.py2rpy(X)
        r_output = drf_r_package.predict_drf(self.r_fit_object, X_r)
        # print(len(r_output))
        # print(type(rpy2.robjects.conversion.ri2py(r_output[0])))
        # print(type(r_output))
        weights = base_r_package.as_matrix(r_output[0])
        y_train = base_r_package.as_matrix(r_output[1])

        if transformation is not None:
            y_train = pd.DataFrame(y_train)
            y_train = y_train.apply(transformation).apply(pd.Series)
            y_train = y_train.values

        available_functionals = {
            "weights": get_weights,
            "mean": get_mean,
            "sd": get_sd,
            "cov": get_cov,
            "cor": get_cor,
            "quantile": get_quantile,
            "sample": get_sample,
        }

        if type(functional) is str:
            return available_functionals[functional](weights, y_train, **predict_kwargs)
        else:
            return_dict = dict.fromkeys(functional)
            for f in functional:
                return_dict[f] = available_functionals[f](weights, y_train, **predict_kwargs)
            return return_dict
