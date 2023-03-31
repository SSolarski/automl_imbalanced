import numpy as np
import pandas as pd

from data import Dataset

from sklearn.model_selection import StratifiedKFold, cross_val_score
from skopt import dump

# This cell runs a small benchmark on the imbalanced datasets
# comparing many different models

from sklearn.metrics import balanced_accuracy_score

from ImbalancedAutoML import ImbalancedAutoMLPipeline

import warnings

# Prevent warnings from printing to console

pd.set_option('display.precision', 2)


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def run_automl_benchmark(data_ids, model_type='brf', outer_folds=3, inner_folds=3, stop_time=60, verbose=0, num_points=1, random_state=42):
    """
    Runs a benchmark on the imbalanced datasets using the automl system

    Parameters
    ----------
    data_ids : list
        List of dataset ids to run the benchmark on
    model_type : str
        The type of model to use in the pipeline. Can be 'brf' or 'xgb'
    outer_folds : int
        The number of outer folds to use in the cross validation
    inner_folds : int
        The number of inner folds to use in the cross validation
    stop_time : int
        The time in seconds to run the automl system for
    verbose : int
        The verbosity of the automl system
    num_points : int
        The number of points to use in the bayesian optimization
    random_state : int
        The random state to use in the bayesian optimization

    Returns
    -------
    all_pipelines_stats : dict
        A dictionary containing the stats of each pipeline

    """

    all_pipelines_stats = dict()

    for id in data_ids:
        dataset = Dataset.from_openml(id)

        print(f"\nRunning Classification on {dataset.name}")
        X = dataset.features
        y = dataset.labels

        outer_acc = list()
        pipeline_stats = dict()

        # configure the cross-validation procedure
        cv_outer = StratifiedKFold(
            n_splits=outer_folds, shuffle=True, random_state=42)

        # outer loop
        for counter, (train_ix, test_ix) in enumerate(cv_outer.split(X, y)):
            print(f"\n  Outer fold #{counter}")

            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            # Initialize the pipeline
            pipeline = ImbalancedAutoMLPipeline(
                model_type=model_type, cv_folds=inner_folds, num_points=num_points,  stop_time=stop_time, verbose=verbose, random_state=random_state)

            # Fit the pipeline
            pipeline.fit(X_train, y_train)

            yhat = pipeline.predict(X_test)
            acc = balanced_accuracy_score(y_test, yhat)

            # Store the results
            outer_acc.append(acc)
            pipeline_stats[f'outer_fold_{counter}'] = {
                "mean_scores": pipeline.tunners['best'].cv_results_['mean_test_score'],
                "std_scores": pipeline.tunners['best'].cv_results_['std_test_score'],
                "best_model_name": pipeline.best_model_name,
                "best_params": pipeline.tunners['best'].best_params_,
                "best_model_acc": pipeline.best_model_acc,
                f"outer_acc_{counter}": acc,
            }

            print(
                f'\n  Evaluation on outer fold {counter}: \n  Accuracy: {acc:.4f}')
            print(
                f'  Estimated accuracy from inner cv: {pipeline.best_model_acc: .4f}')

            # Save the current pipeline (takes a lot of space)
            #dump(pipeline, f'saved/pipeline_{id}_{counter}.pkl')

        # summarize the estimated performance of the model
        print(
            f'\nAverage outer CV Accuracy: {np.mean(outer_acc):.4f}, std: ({np.std(outer_acc):.4f})')

        # Save the accuracy of the dataset
        pipeline_stats['outer_acc'] = np.mean(outer_acc)
        all_pipelines_stats[id] = pipeline_stats

        # Save the stats of the pipeline
        dump(pipeline_stats, f'saved/pipeline_stats_{id}.pkl')

    # Save the stats of all the pipelines together
    dump(all_pipelines_stats, 'saved/all_pipelines_stats.pkl')


def run_default_model_benchmark(data_ids, models, scores_dict, scoring):
    for id in data_ids:
        dataset = Dataset.from_openml(id)

        print(f"\nRunning Classification tree on {dataset.name}")

        X = dataset.features
        y = dataset.labels

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for model in models:
            scores = cross_val_score(
                models[model], X, y, scoring=scoring, cv=cv)
            print(f"Balanced Accuracy of {model}: {np.mean(scores)}")

            scores_dict[model].append(np.mean(scores))

        dump(scores_dict, f'saved/scores_dict.pkl')


def dataframe_from_scores_dict(scores_dict, data_ids):
    """
    Creates a dataframe from the scores dictionary
    """
    scores_df = pd.DataFrame(scores_dict)
    scores_df.index = data_ids
    return scores_df


def display_scores(scores_df):
    """
    Displays the scores dataframe in a nice format
    """
    display(scores_df.T.style.highlight_max(color='lightgreen', axis=0))
