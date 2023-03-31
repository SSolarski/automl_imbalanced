from sklearn import impute

from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from xgboost import XGBClassifier

# This file holds all the base models that are used in the benchmark

models = {}

models["Decision Tree"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", DecisionTreeClassifier(random_state=42)),
    ]
)
models["Random Forest"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", RandomForestClassifier(random_state=42)),
    ]
)
models["XGBoost"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", XGBClassifier(random_state=42)),
    ]
)

models["XGBoost SMOTE"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("resampling", SMOTE(random_state=42)),
        ("estimator", XGBClassifier(random_state=42)),
    ]
)
models["XGBoost SMOTETomek"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("resampling", SMOTETomek(random_state=42)),
        ("estimator", XGBClassifier(random_state=42)),
    ]
)

models["XGBoost RandomUnderSampler"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("resampling", RandomUnderSampler(random_state=42)),
        ("estimator", XGBClassifier(random_state=42)),
    ]
)
models["Random Forest SMOTE"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ('resampling', SMOTE(random_state=42)),
        ("estimator", RandomForestClassifier(random_state=42)),
    ]
)


models["XGBoost Weighted"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", XGBClassifier(scale_pos_weight=99, random_state=42)),
    ]
)
models["Random Forest Weighted"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", RandomForestClassifier(
            class_weight='balanced', random_state=42)),
    ]
)

models["RUSBoost"] = imb_pipeline(
    steps=[
        ('imputer', impute.SimpleImputer(strategy='median')),
        ('rusboost', RUSBoostClassifier(random_state=42))
    ]
)

models["Balanced Random Forest"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", BalancedRandomForestClassifier(random_state=42)),
    ]
)

models["Balanced Random Forest KNNImputer"] = imb_pipeline(
    steps=[
        ("imputer", impute.KNNImputer(n_neighbors=5)),
        ("estimator", BalancedRandomForestClassifier(random_state=42)),
    ]
)
models["XGBoost RandomUnderSampler KNNImputer"] = imb_pipeline(
    steps=[
        ("imputer", impute.KNNImputer(n_neighbors=5)),
        ("resampling", RandomUnderSampler(random_state=42)),
        ("estimator", XGBClassifier(random_state=42)),
    ]
)

models["Voting Classifier"] = imb_pipeline(
    steps=[
        ("imputer", impute.SimpleImputer()),
        ("estimator", VotingClassifier(estimators=[
            ('brf', models["Balanced Random Forest"]),
            ('xgb', models["XGBoost RandomUnderSampler"]),
        ],
            voting='soft',)),
    ]
)


scores_dict = {}
for model in models:
    scores_dict[model] = []
