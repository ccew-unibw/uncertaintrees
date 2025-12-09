# Forests of UncertainT(r)ees: Using Tree-based Ensembles to Estimate Probability Distributions of Future Conflict

This repository contains the replication code for *Forests of UncertainT(r)ees: Using Tree-based Ensembles to Estimate Probability Distributions of Future Conflict*.

## Abstract
_Predictions of fatalities from violent conflict on the PRIO-GRID-month (pgm) level are characterized by high levels of uncertainty, limiting their usefulness in practical applications.
We discuss the two main sources of uncertainty for this prediction task, the nature of violent conflict and data limitations, embedding this in the wider literature on uncertainty
quantification in machine learning. We develop a strategy to quantify uncertainty in conflict predictions yielding samples from a predictive distribution, allowing us to estimate
prediction intervals. Our approach compares and combines multiple tree-based classifiers and distributional regressors in a custom auto-ML setup, estimating distributions for each
pgm individually. We also test the integration of regional models in spatial ensembles as a potential avenue to reduce uncertainty. The models are able to consistently outperform a
suite of benchmarks derived from conflict history in predictions up to one year in advance, with performance driven by regions where conflict was observed. With our evaluation, we
emphasize the need to understand how a metric behaves for a given prediction problem, in our case characterized by extremely high zero-inflatedness. While not resulting in better
predictions, the integration of smaller models does not decrease performance for this prediction task, opening avenues to integrate data sources with less spatial coverage in the future._

### VIEWS challenge

The paper is based on a submission to the [ViEWS Prediction Challenge 2023](https://viewsforecasting.org/research/prediction-challenge-2023/). For more information, see

Hegre et al. (2025). The 2023/24 ViEWS prediction challenge: Predicting the number of fatalities in armed conflict, with uncertainty. _Journal of Peace Research_ 62(6): 2070–2087.


## Usage

> :warning: **This repo uses git-submodules!** Run `git submodule init` and  `git submodule update` after cloning or clone with `git clone --recurse-submodules`. The official [ViEWS Prediction Challenge repo](https://github.com/prio-data/prediction_competition_2023) is linked into this repo as a submodule and required for the evaluation scripts to run.

To download the data provided by the ViEWS team, run `./download_data.sh` if you are operating a Unix-like system or `./download_data.ps1` if you are on Windows. The download script also downloads the [CGAZ ADM0 GeoBoundaries](https://www.geoboundaries.org/globalDownloads.html) dataset used for creating the UN-region based clusters. You can then either run the whole (i) _training and prediction_ as well as the (ii) _evaluation_ pipelines directly on your machine using Python and a virtual environment or you can use our Docker image to run it (see below).

At the heart of the project is the _tuning and prediction pipeline_, which is defined and controlled by the `competition_pipeline.py` file. You can adjust any setting related to the initial training and the generation of prediction directly in `competition_pipeline.py`. After producing predictions based on the training runs, `evaluation_pipeline.py` evaluates the predictions against the metrics and benchmark models defined in the invitation to the [ViEWS Prediction Challenge 2023](https://viewsforecasting.org/research/prediction-challenge-2023/). As with the _tuning and prediction pipeline_, settings can be adjusted in the script directly. 

The `views_evaluation.ipynb` Jupyter notebook is the basis of our evaluation, and is used to pruduce the figures for the paper while providing a few additional insights.


### Structure

Our prediction pipeline represents a custom AutoML approach to combining tree-based classifiers and distributional regressors in a quasi-hurdle approach, creating forecasts for 3 to 14 months into the future.

Included estimators are:

- For the binary classification step:
    - [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    - [XGBoost](https://github.com/dmlc/xgboost)

- For the regression step:
    - [NGBoost](https://github.com/stanfordmlgroup/ngboost)
    - [Distributional Random Forests](https://github.com/lorismichel/drf)
    - [Quantile Regression Forests](https://github.com/zillow/quantile-forest)


The different components needed to run the pipeline are structured into estimators and utils.

- `src/estimators` includes sklearn-compatible wrappers for the Distributional Random Forest and the NGBoost implementations, as well as our hurdle class, which handles tuning, model selection and generation of raw predictions, and our global-local ensemble class.
- `src/utils` contains all workhorse functions called by the two pipelines to load data, handle conversions, create clusters, tune models, score models and handle evaluations.

The following outputs are stored in the repository:

- `modelling/` contains the hyperopt trials objects from hyperparameter tuning, which store tuning performance and are used to select the best models, as well as raw predictions from the two hurdle steps. It is also used to store temporary prediction files for individual timesteps during a pipeline run.
- `evaluation/` contains calculated evaluation metrics for our models and the five benchmarks, both globally and for individual countries/clusters.
- `submissions/` contains the directories with the final predictions for the respective model specifications.
- `figures/` contains the figures generated with the `views_evaluation.ipynb` notebook for the preprint.

The required inputs for clustering and plotting are stored in `data/`.

NOTE: While this pipeline can be used to produce country-level models as well, we did not implement any data cleaning and matching regarding changes in the list of countries over time, as our focus was on the pgm-models.

### Requirements

- Docker (if you want to run everything in Docker)
- Python 3.11 and R (if you want to run the code directly on your machine)
- ~ 64GB memory

Running the `competition_pipeline.py` took 1.5-2 weeks on a (somewhat dated) 40 CPU / 128GB memory setup.


### a) Vanilla Python

#### Unix-like systems

- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `python competition_pipeline.py` or `./run_pipeline.sh`
- `python evaluation_pipeline.py` or `./run_pipeline.sh evaluate` for the evaluation pipeline

#### Windows

The vanilla Python method is untested on Windows and will potentially fail. Use the Docker image instead (see below).

### b) Docker

You can either use the provided .sh/.ps1 script to start the docker container by adding the -d flag to `./run_pipeline.sh` (or by running `run_pipeline start` on Windows) or run the docker commands directly.

#### Unix-like systems

- `./run_pipeline.sh -d` or `./run_pipeline.sh -d predict` to start the container and run the pipeline. If you add the `-l` flag, the container still starts in the background but you will directly peek into the logs.
- `./run_pipeline.sh -d evaluate` runs the evaluation pipeline
- `./run_pipeline.sh logs` to peek into the logs of the running container
- `./run_pipeline.sh stop` to stop the running container

You can also do all of this manually by directly calling the required docker commands.

- `docker build -t ccew-tree .`
- `docker run -d -v .:/usr/src/app ccew-tree "python3.11 -u competition_pipeline.py"` or `docker run -d -v .:/usr/src/app ccew-tree "python3.11 -u evaluation_pipeline.py"`

#### Windows

- `.\run_pipeline.ps1 predict` to start the container and run the pipeline. If you add the `-l` flag, the container still starts in the background but you will directly peek into the logs.
- `.\run_pipeline.ps1 evaluate` to run the evaluation pipeline
- `.\run_pipeline.ps1 logs` to peek into the logs of the running container
- `.\run_pipeline.ps1 stop` to stop the running container

You can also do all of this manually by directly calling the required docker commands.

- `docker build -t ccew-tree .`
- `docker run -d -v ${PWD}:/usr/src/app ccew-tree "python3.11 -u competition_pipeline.py"` or `docker run -d -v .:/usr/src/app ccew-tree "python3.11 -u evaluation_pipeline.py"`

### Test mode

To quickly test the pipeline on your machine, you can append the `-t` flag either to the bash script or when running the `competition_pipeline.py` directly:

#### Unix-like systems

- `./run_pipeline.sh -t` (local run)
- `./run_pipeline.sh -d -t` (Docker run)
- `python competition_pipeline.py -t`

#### Windows

- `.\run_pipeline.ps1 predict -t`

## Reference
When you are using (parts of) our work, please cite:

*Mittermaier, D., Bohne, T., Hofer, M. & Racek, D. (2025). Forests of UncertainT(r)ees: Using Tree-based Ensembles to Estimate Probability Distributions of Future Conflict. https://arxiv.org/abs/2512.06210.*

## Contributing
We welcome contributions to enhance the models and methodologies used in this study. Please submit pull requests or open issues for any suggestions or improvements.

## Authors
- [Daniel Mittermaier](mailto:daniel.mittermaier@unibw.de)
- Tobias Bohne
- Martin Hofer
- Daniel Racek
