import pickle
from os.path import exists

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

paths_lists = [
    "/home/mahmoudk/iris/dataset/LBP/r_1.pkl",
    "/home/mahmoudk/iris/dataset/LBP/r_3.pkl",
    "/home/mahmoudk/iris/dataset/LBP/r_5.pkl",
    "/home/mahmoudk/iris/dataset/LBP/concat_lbp.pkl",
    "/home/mahmoudk/iris/dataset/MB_LBP/h_3_w_5.pkl",
    "/home/mahmoudk/iris/dataset/MB_LBP/h_2_w_10.pkl",
    "/home/mahmoudk/iris/dataset/HOG/hog.pkl"
    # "/home/mahmoudk/iris/dataset/HOG/hog_4_(6, 10).pkl",
    # "/home/mahmoudk/iris/dataset/HOG/hog_8_(6, 10).pkl",
    # "/home/mahmoudk/iris/dataset/HOG/hog_16_(6, 10).pkl"
]

results_path = "/home/mahmoudk/iris/dataset/results_without_scale.pkl"
SEED = 42
N_JOBS = 20
SPLITS = 5
TEST_SIZE = 0.5

# Vanilla Logistic Regression
gv_model_logistic_regression = GridSearchCV(
    LogisticRegression(
        random_state=SEED,
        class_weight="balanced",
        solver="liblinear",
        max_iter=300
    ), {
        "C": [0.1, 1., 10, 50],
        "penalty": ["l1", "l2"]
    },
    cv=StratifiedKFold(n_splits=SPLITS),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose=10,
    n_jobs=N_JOBS
)

# Logistic Regression with PCA
gv_model_logistic_regression_with_pca = GridSearchCV(
    Pipeline([('scalar', StandardScaler()),
              ('pca', PCA()),
              ('lg', LogisticRegression(
                  random_state=SEED,
                  class_weight="balanced",
                  solver="liblinear",
                  max_iter=300
              ))]),
    {
        "pca__n_components": [10, 50, 100, 500, 1000, 1400],
        "lg__C": [0.1, 1, 10, 50],
        "lg__penalty": ["l1", "l2"]
    },
    cv=StratifiedKFold(n_splits=SPLITS),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose=10,
    n_jobs=N_JOBS
)

# Logistic Regression with selection
gv_model_logistic_regression_with_selection = GridSearchCV(
    Pipeline([('scalar', StandardScaler()),
              ('feature_selection', SelectKBest(mutual_info_classif)),
              ('lg', LogisticRegression(
                  random_state=SEED,
                  class_weight="balanced",
                  solver="liblinear",
                  max_iter=300
              ))]),
    {
        "feature_selection__k": [10, 100, 500, 1000, 1500],
        "lg__C": [0.1, 1, 10, 50],
        "lg__penalty": ["l1", "l2"]
    },
    cv=StratifiedKFold(n_splits=SPLITS),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose=10,
    n_jobs=N_JOBS
)

gv_model_random_forest = GridSearchCV(
    RandomForestClassifier(
        random_state=SEED,
        class_weight='balanced',
    ),
    {
        "n_estimators": [1000, 2000],
        "max_depth": [20],
    },
    cv=StratifiedKFold(n_splits=SPLITS),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose=10,
    n_jobs=N_JOBS
)

gv_model_random_forest_rfe = GridSearchCV(
    RFE(RandomForestClassifier(
        random_state=SEED,
        class_weight='balanced',
        n_estimators=500,
        max_depth=10
    ), verbose=10),
    {
        "n_features_to_select": [10, 100, 500],
        "step": [0.1],
    },
    cv=StratifiedKFold(n_splits=SPLITS),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose=10,
    n_jobs=N_JOBS
)

gv_model_knn = GridSearchCV(
    KNeighborsClassifier(),
    {
        'n_neighbors': [1, 3, 5]
    },
    cv=StratifiedKFold(n_splits=SPLITS),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose=10,
    n_jobs=N_JOBS,
)

gv_model_adaboost = GridSearchCV(
    AdaBoostClassifier(
        random_state=SEED,
    ),
    {
        "n_estimators": [400, 700],
        "learning_rate": [0.5, 1]
    },
    cv=StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose=10,
    n_jobs=N_JOBS
)


gv_GB = GridSearchCV(
    Pipeline([('scaler', StandardScaler()),
              ('GB', GradientBoostingClassifier())]),
    {
        "GB__loss":["deviance"],
        "GB__learning_rate": [0.15],
        "GB__min_samples_split": [3],
        "GB__min_samples_leaf": [0.1],
        "GB__max_depth":[10],
        "GB__max_features":["sqrt"],
        "GB__criterion": ["mse"],
        "GB__subsample":[0.8],
        "GB__n_estimators":[50]
    },
    cv=StratifiedKFold(n_splits=SPLITS),
    scoring='roc_auc_ovr',
    error_score=0,
    verbose = 10,
    n_jobs=N_JOBS
)

algorithms = {
    'lg': gv_model_logistic_regression,
    'lg_pca': gv_model_logistic_regression_with_pca,
    'lg_selection': gv_model_logistic_regression_with_selection,
    'rf': gv_model_random_forest,
    'rf_rfe': gv_model_random_forest_rfe,
    'knn': gv_model_knn,
    'ada': gv_model_adaboost,
    'gb': gv_GB
}

if exists(results_path):
    results = pickle.load(open(results_path, 'rb'))
else:
    results = {}

for key, gv_model in algorithms.items():
    print(f"Working on algorithm {key}")
    for path in paths_lists:

        if f"{key}_{path}" in results.keys():
            continue

        print(f"Working on the path: {path}")
        data = pickle.load(open(path, 'rb'))
        features_list = data['features']
        labels_list = data['labels']

        X = np.array(features_list)
        y = np.array(labels_list)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)

        gv_model.fit(X_train, y_train)

        print('-----')
        print(f'Best parameters {gv_model.best_params_}')
        print(
            f'Mean cross-validated accuracy score of the best_estimator: ' +
            f'{gv_model.best_score_:.3f}'
        )
        y_pred = gv_model.predict(X_test)
        y_pred_prob = gv_model.predict_proba(X_test)
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
        auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        print(f"F1 score: {f1}")
        print(f"ROC: {auc}")
        print('-----')

        results[f"{key}_{path}"] = {"train_auc": gv_model.best_score_, "test_auc": auc,
                                    "best_params": gv_model.best_params_}

        pickle.dump(results, open(results_path, 'wb'))




