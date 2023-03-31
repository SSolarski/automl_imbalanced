from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer, KNNImputer
from skopt.space import Real, Integer, Categorical

data_ids = (976, 980, 1002, 1018, 1019, 1021, 1040, 1053, 1461, 41160)

scoring = "balanced_accuracy"

# Define the search space for Balanced Random Forest model
brf_search_spaces = {
    'classifier__n_estimators': Integer(100, 1000),
    'classifier__max_depth': Integer(1, 50),
    'classifier__min_samples_split': Integer(2, 10),
    'classifier__min_samples_leaf': Integer(1, 10),
    'classifier__ccp_alpha': Real(0.0, 0.05, 'uniform'),
    'feature_selection__threshold': Real(0, 1),
    'preprocessor': Categorical([SimpleImputer(), KNNImputer(n_neighbors=5)]),
}

# Define the search space for the XGBoost model
xgb_search_spaces = {
    'classifier__n_estimators': Integer(100, 1000),
    'classifier__max_depth': Integer(3, 10),
    'classifier__learning_rate': Real(0.001, 0.1, 'log-uniform'),
    'classifier__gamma': Real(0.1, 1.0, 'uniform'),
    'classifier__reg_lambda': Real(0.0, 1.0, 'uniform'),
    'classifier__subsample': Real(0.5, 1.0, 'uniform'),
    'feature_selection__threshold': Real(0, 1),
    'resampling': [SMOTE(), SMOTETomek(), RandomUnderSampler()],
    'preprocessor': Categorical([SimpleImputer(), KNNImputer(n_neighbors=5)]),
}

# Define the search space for the Decision Tree model
dt_search_spaces = {
    'classifier__max_depth': Integer(5, 50),
    'classifier__min_samples_split': Integer(2, 10),
    'classifier__min_samples_leaf': Integer(1, 10),
    'classifier__ccp_alpha': Real(0.0, 0.1),
    'feature_selection__threshold': Real(0, 1),
    'resampling': [SMOTE(), SMOTETomek(), RandomUnderSampler()],
    'preprocessor': Categorical([SimpleImputer(), KNNImputer(n_neighbors=5)]),
}

# Define the search space for the CatBoost model
cb_search_spaces = {
    'classifier__iterations': Integer(50, 500),
    'classifier__learning_rate': Real(0.01, 0.5, prior='log-uniform'),
    'classifier__depth': Integer(1, 16),
    'classifier__l2_leaf_reg': Integer(1, 10),
    'classifier__border_count': Integer(32, 255),
    'classifier__bagging_temperature': Real(0, 10),
    'classifier__random_strength': Real(0, 10),
    'classifier__scale_pos_weight': Real(0, 10),
    'classifier__one_hot_max_size': Integer(2, 16),
    'classifier__max_ctr_complexity': Integer(1, 10),
    'classifier__fold_permutation_block': Integer(1, 10),
    'classifier__leaf_estimation_iterations': Integer(1, 10),
    'classifier__subsample': Real(0.1, 1, prior='log-uniform'),
    'feature_selection__threshold': Real(0, 1),
    'resampling': [SMOTE(), SMOTETomek(), RandomUnderSampler()],
    'preprocessor': Categorical([SimpleImputer(), KNNImputer(n_neighbors=5)]),
}


# Define the search space the ensemble model
ensemble_search_spaces = {
    'classifier__brf__n_estimators': Integer(100, 1000),
    'classifier__brf__max_depth': Integer(5, 50),
    'classifier__brf__min_samples_split': Integer(2, 10),
    'classifier__brf__min_samples_leaf': Integer(1, 10),
    'classifier__brf__ccp_alpha': Real(0.0, 0.1),
    'classifier__xgb__n_estimators': Integer(100, 1000),
    'classifier__xgb__max_depth': Integer(3, 10),
    'classifier__xgb__learning_rate': Real(0.001, 0.1, 'log-uniform'),
    'classifier__xgb__gamma': Real(0.1, 1.0, 'uniform'),
    'classifier__xgb__reg_lambda': Real(0.0, 1.0, 'uniform'),
    'classifier__xgb__subsample': Real(0.5, 1.0, 'uniform'),
    'resampling': [None, SMOTE(), SMOTETomek(), RandomUnderSampler()],
    'feature_selection__threshold': Real(0, 1),
    'preprocessor': Categorical([SimpleImputer(), KNNImputer(n_neighbors=5)]),
    'classifier__voting': ['soft', 'hard'],
}
