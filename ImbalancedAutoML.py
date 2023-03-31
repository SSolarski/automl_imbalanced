import warnings

import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold

from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback

from xgboost import XGBClassifier

import configuration as config


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class ImbalancedAutoMLPipeline(BaseEstimator, ClassifierMixin):
    """Imbalanced AutoML Pipeline class.

    This class implements an imbalanced AutoML pipeline that uses a
    cross-validated Bayesian Optimization search to find the best hyperparameters for
    a given dataset. It returns the best model found during the search.
    It takes implements the following steps:

    1. Preprocessing: Missing values are imputed using the mean value of the column.
    2. Resampling: The dataset is resampled using SMOTE-Tomek.
    3. Feature selection: Constant features are dropped.
    4. Classification: The best model is selected using a cross-validated Bayesian
    Optimization search.    

    Parameters:
    -----------
    model_type (str, default='auto'): The type of model(s) to fit. 
        Possible values are 'auto', 'ensemble', 'brf', 'dt', and 'xgb'.
    cv_folds (int, default=3): The number of folds to use in the cross-validation.
    num_points (int, default=4): The number of points to sample in the search space at every iteration.
    stop_time (int, default=3600): The time in seconds to stop the tuning.
    verbose (int, default=0): Whether to print the progress of the tuning.
    random_state (int, default=42): Random state for reproducing results.

    Attributes:
    -----------
    classifiers (dict): The classifiers for each model.
    models (dict): The models (pipelines) including preprocessing, resampling and classifier.
    estimators (dict): The fitted estimators for each model.
    tunners (dict): The tunners for each model.
    best_model_acc (float): The accuracy of the best model.
    best_model_name (str): The name of the best model.

    Note: Before calling the fit() method, the config module must be imported to load the 
          search spaces for the hyperparameters of each classifier.

    """

    def __init__(self, model_type='auto', cv_folds=3, num_points=4, stop_time=60*60*1, verbose=0, random_state=42):
        """Initialize the ImbalancedAutoMLPipeline class."""

        self.model_type = model_type  # The type of model(s) to fit
        self.cv_folds = cv_folds  # The number of folds to use in the cross-validation
        # The number of points to sample in the search space at every iteration
        self.num_points = num_points
        self.stop_time = stop_time  # The time in seconds to stop the tuning
        self.verbose = verbose  # Whether to print the progress of the tuning
        self.random_state = random_state  # Random state for reproducing results
        self.classifiers = dict()  # The classifiers for each model
        # The models (pipelines) including preprocessing, resampling and classifier
        self.models = dict()
        self.estimators = dict()  # The fitted estimators for each model
        self.tunners = dict()  # The tunners for each model
        self.best_model_acc = None  # The accuracy of the best model
        self.best_model_name = None  # The name of the best model

        assert self.model_type in ['auto', 'all', 'ensemble', 'brf',
                                   'dt', 'xgb'], 'model_type must be one of [auto, all, ensemble, brf, xgb, dt]'
        # Define the preprocessor
        self.preprocessor = SimpleImputer()

        # Drop all constant features
        self.feature_selection = VarianceThreshold(threshold=0)

        # Define the initial resampling strategy
        self.resampling = SMOTETomek(random_state=self.random_state)

        # Define the classifiers with default hyperparameters
        self.brf_classifier = BalancedRandomForestClassifier(
            random_state=self.random_state)

        self.xgb_classifier = XGBClassifier(
            random_state=self.random_state)

        self.dt_classifier = DecisionTreeClassifier(
            random_state=self.random_state)

        # self.cb_classifier = CatBoostClassifier(
        #     random_state=self.random_state, verbose=False)

        self.ensemble_classifier = VotingClassifier(
            estimators=[('brf', self.brf_classifier),
                        ('xgb', self.xgb_classifier)],
            voting='soft')

        # Take the search spaces from the config for each classifier
        self.brf_search_spaces = config.brf_search_spaces
        self.dt_search_spaces = config.dt_search_spaces
        self.xgb_search_spaces = config.xgb_search_spaces
        # self.cb_search_spaces = config.cb_search_spaces

        self.ensemble_search_spaces = config.ensemble_search_spaces

    def fit(self, X, y):
        """
        Fits the models and returns the best model based on balanced accuracy.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target labels.

        Returns:
        --------
        self : object
            Returns self.

        """

        # Get the classifiers, models and tunners
        self.get_classifiers()
        self.get_tuners_and_models(self.cv_folds, self.num_points)

        # Empty list to store the callbacks
        callback_list = []

        print("    Started fitting the models")

        # Empty dictionary to store the best models
        self.estimators = {}

        # Variables to store the best results
        best_model_acc = -1
        best_tunner_acc = -1
        best_model = None
        best_tunner = None

        # Branch to choose which model to optimize

        if self.model_type == 'auto':
            print("    Checking the default models to choose which one to optimize")
            for name, tunner in self.tunners.items():
                print(f"\n      Fitting {name} model")

                cv = StratifiedKFold(n_splits=self.cv_folds,
                                     shuffle=True, random_state=42)
                scores_model = cross_val_score(
                    self.models[name], X, y, scoring=config.scoring, cv=cv).mean()
                print(f"      Default score for {name} model: {scores_model}")

                # Check if the model is better than the previous one
                if scores_model > best_model_acc:
                    best_model_acc = scores_model
                    best_model = self.models[name]
                    self.best_model_name = name
                    self.models[name].fit(X, y)

            # Once the best model is found, optimize it
            print(f"\n    Trying to optimize the {self.best_model_name} model")
            print(
                f"    This should take around {np.ceil(self.stop_time/60)} minutes")

            # Define the callbacks
            if self.verbose != 0:
                verbose_callback = VerboseCallback(n_total=self.verbose)
                callback_list.append(verbose_callback)
            stopper_callback = DeadlineStopper(self.stop_time)
            callback_list.append(stopper_callback)

            # Fit the tunner for the best model
            tunner = self.tunners[self.best_model_name]
            tunner.fit(
                X, y, callback=callback_list)

            print(
                f"      Best score for tunned {self.best_model_name} model: {tunner.best_score_:.4f}")
            self.estimators[self.best_model_name] = tunner.best_estimator_

            # Save the best model and result
            if tunner.best_score_ > best_tunner_acc:
                best_tunner_acc = tunner.best_score_
                best_tunner = tunner

            if tunner.best_score_ > best_model_acc:
                best_tunner = tunner
                best_model_acc = best_tunner.best_score_
                best_model = best_tunner.best_estimator_

        # If the model type is not auto, optimize the specific chosen model
        else:
            for name, tunner in self.tunners.items():
                print(f"\n      Fitting {name} model")

                # Run the default models
                cv = StratifiedKFold(n_splits=self.cv_folds,
                                     shuffle=True, random_state=42)
                scores_model = cross_val_score(
                    self.models[name], X, y, scoring=config.scoring, cv=cv).mean()
                print(f"      Default score for {name} model: {scores_model}")

                # Check if the model is better than the previous one
                if scores_model > best_model_acc:
                    best_model_acc = scores_model
                    best_model = self.models[name]
                    self.best_model_name = name
                    self.models[name].fit(X, y)

                # Define the callbacks
                if self.verbose != 0:
                    verbose_callback = VerboseCallback(n_total=self.verbose)
                    callback_list.append(verbose_callback)

                # Divide the stop time by the number of models to optimize
                stop_time = self.stop_time/len(self.tunners)
                stopper_callback = DeadlineStopper(stop_time)
                callback_list.append(stopper_callback)

                print(
                    f"\n      Trying to optimize the {self.best_model_name} model")
                print(
                    f"      This should take around {np.ceil(stop_time/60)} minutes")
                tunner.fit(
                    X, y, callback=callback_list)

                print(
                    f"      Best score for tunned {name} model: {tunner.best_score_:.4f}")

                self.estimators[name] = tunner.best_estimator_

                # Save the best tunner
                if tunner.best_score_ > best_tunner_acc:
                    best_tunner_acc = tunner.best_score_
                    best_tunner = tunner

                # Save the best model and result
                if tunner.best_score_ > best_model_acc:
                    best_tunner = tunner
                    best_model_acc = best_tunner.best_score_
                    best_model = best_tunner.best_estimator_

        # Add the best estimator and tunner to the dictionary
        self.best_model_acc = best_model_acc
        self.estimators['best'] = best_model
        self.tunners['best'] = best_tunner

        # fit the best model on the whole training data
        self.estimators['best'].fit(X, y)

        return self

    def fit_best_estimator(self, X, y):
        """
        Fits the best model on the whole training data.
        Used to convert previously piclke pipelines to the new format.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target labels.
        """

        # Fit the best estimator on the whole training data
        self.estimators['best'].fit(X, y)

    def predict(self, X):
        """
        Predicts the labels for the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            The predicted labels.
        """

        # Use the best estimator to predict results
        return self.estimators['best'].predict(X)

    def get_classifiers(self):
        """
        Gets the classifiers based on model_type.
        """

        # Decide which classifiers to use
        if (self.model_type == 'auto' or self.model_type == 'all'):
            self.classifiers.update(
                {'BalancedRandomForest': (self.brf_classifier, self.brf_search_spaces)})
            self.classifiers.update(
                {'DecisionTree': (self.dt_classifier, self.dt_search_spaces)})
            self.classifiers.update(
                {'XGBoost': (self.xgb_classifier, self.xgb_search_spaces)})
            # self.classifiers.update(
            #     {'CatBoost': (self.cb_classifier, self.cb_search_spaces)})
            self.classifiers.update(
                {'Ensemble': (self.ensemble_classifier, self.ensemble_search_spaces)})
        elif self.model_type == 'brf':
            self.classifiers.update(
                {'BalancedRandomForest': (self.brf_classifier, self.brf_search_spaces)})

        elif self.model_type == 'dt':
            self.classifiers.update(
                {'DecisionTree': (self.dt_classifier, self.dt_search_spaces)})

        elif self.model_type == 'xgb':
            self.classifiers.update(
                {'XGBoost': (self.xgb_classifier, self.xgb_search_spaces)})

        # elif self.model_type == 'cb':
        #     self.classifiers.update(
        #         {'CatBoost': (self.cb_classifier, self.cb_search_spaces)})

        elif self.model_type == 'ensemble':
            self.classifiers.update(
                {'Ensemble': (self.ensemble_classifier, self.ensemble_search_spaces)})

    def get_tuners_and_models(self, cv_folds=3, num_points=4):
        """
        Gets the tuners and models to be used in the model.
        """

        for clf_name, (clf, search_space) in self.classifiers.items():

            # If the model is the BalancedRandomForest, don't use resampling
            if clf_name == 'BalancedRandomForest':
                model = imb_pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('feature_selection', self.feature_selection),
                    ('resampling', None),
                    ('classifier', clf),
                ])
            # if the model is the XGBoost, use the RandomUnderSampler
            elif clf_name == 'XGBoost':
                model = imb_pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('feature_selection', self.feature_selection),
                    ('resampling', RandomUnderSampler(random_state=42)),
                    ('classifier', clf),
                ])
            else:
                model = imb_pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('feature_selection', self.feature_selection),
                    ('resampling', self.resampling),
                    ('classifier', clf),
                ])

            cv = StratifiedKFold(n_splits=cv_folds,
                                 shuffle=True, random_state=42)

            # Define the BayesSearchCV for each model
            bayes_cv_tuner = BayesSearchCV(
                estimator=model,
                search_spaces=search_space,
                scoring=config.scoring,
                n_iter=100000,
                cv=cv,
                verbose=0,
                n_jobs=-1,
                n_points=num_points,
                random_state=42,
                return_train_score=True,
            )
            self.models[clf_name] = model
            self.tunners[clf_name] = bayes_cv_tuner
