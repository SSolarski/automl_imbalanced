# AutoML pipeline for imabalanced classification

A project for the AutoML course WS22/23 by Stefan Solarski at LMU.

## Contents

The ImbalancedAutoMLPipeline class is in the [ImbalancedAutoML.py](https://github.com/SSolarski/automl_imbalanced/blob/7cd18d49a254414d3b69790a5b0c0b4a4f4aff65/ImbalancedAutoML.py) file and it is where the most important code is. It reads hyperparameter search spaces from the [configuration.py](https://github.com/SSolarski/automl_imbalanced/blob/7cd18d49a254414d3b69790a5b0c0b4a4f4aff65/configuration.py) file. 

The Jupyter [notebook](https://github.com/SSolarski/automl_imbalanced/blob/7cd18d49a254414d3b69790a5b0c0b4a4f4aff65/notebook.ipynb) is used to run the benchmarks for the automl system, as well as the vizualizations and initial benchmarking of default parameter classifiers. 

The final [report](https://github.com/SSolarski/automl_imbalanced/blob/7cd18d49a254414d3b69790a5b0c0b4a4f4aff65/Stefan%20Solarski%20Report%20AutoML.pdf) explains the approaches used to tackle the problem of imbalanced learning, the experiments we conducted, the resulting pipeline, and results.

## Setup

Clone the repository and use [pip](https://pip.pypa.io/en/stable/), or another package manager, to install the requirements.

```bash
git clone https://github.com/SSolarski/automl_imbalanced.git
cd automl_imbalanced
pip install -r requirements.txt
```

Necessary packages are given in [requirements.txt](https://github.com/SSolarski/automl_imbalanced/blob/7cd18d49a254414d3b69790a5b0c0b4a4f4aff65/requirements.txt), we used Python v3.8.16.

## Packages

We used the following packages:

1. [pandas](https://pandas.pydata.org/docs/index.html) -> manipulating and displaying the datasets and results
1. [jinja2](https://jinja.palletsprojects.com/en/3.1.x/) -> improves the style of pandas dataframes
1. [numpy](https://numpy.org/doc/) -> numerical calculations
1. [openml](https://docs.openml.org/) -> importing datasets from openml
1. [scikit-learn](https://scikit-learn.org/stable/user_guide.html) -> basic classifiers, pipelines and preprocessing
1. [imbalanced-learn](https://imbalanced-learn.org/stable/) -> classifiers, pipelines and preprocessing for imbalanced datasets
1. [xgboost](https://xgboost.readthedocs.io/en/stable/) -> XGBoost classifier
1. [scikit-optimize](https://scikit-optimize.github.io/stable/) -> hyperparameter tuning using Bayesian optimization
1. [matplotlib](https://matplotlib.org/stable/index.html) -> visualizing performance of the automl system
