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

from metrics import EER

paths_lists = [
    "/home/mahmoudk/iris/dataset/LBP/r_1.pkl",
    "/home/mahmoudk/iris/dataset/LBP/r_3.pkl",
    "/home/mahmoudk/iris/dataset/LBP/r_5.pkl",
    "/home/mahmoudk/iris/dataset/LBP/concat_lbp.pkl",
    "/home/mahmoudk/iris/dataset/MB_LBP/h_3_w_5.pkl",
    "/home/mahmoudk/iris/dataset/MB_LBP/h_2_w_10.pkl",
    "/home/mahmoudk/iris/dataset/HOG/hog_4_(6, 10).pkl",
    "/home/mahmoudk/iris/dataset/HOG/hog_8_(6, 10).pkl",
    "/home/mahmoudk/iris/dataset/HOG/hog_16_(6, 10).pkl"
]

i = 8
path = paths_lists[i]
results_path = f"/home/mahmoudk/iris/dataset/eer_f{i}"
SEED = 42
N_JOBS = 30
TEST_SIZE = 0.5

# Vanilla Logistic Regression
lg_model = LogisticRegression(
    random_state=SEED,
    class_weight="balanced",
    solver="liblinear",
    max_iter=300,
    verbose=False,
    n_jobs=N_JOBS,
    C=50,
    penalty='l2'
)

lg_pca = Pipeline([('scalar', StandardScaler()),
                   ('pca', PCA(n_components=100)),
                   ('lg', LogisticRegression(
                       random_state=SEED,
                       class_weight="balanced",
                       solver="liblinear",
                       max_iter=300,
                       n_jobs=N_JOBS,
                       C=50,
                       penalty='l1'
                   ))], verbose=True)

lg_feature_selection = Pipeline([('scalar', StandardScaler()),
                                 ('feature_selection', SelectKBest(mutual_info_classif, k=1500)),
                                 ('lg', LogisticRegression(
                                     random_state=SEED,
                                     class_weight="balanced",
                                     solver="liblinear",
                                     max_iter=300,
                                     n_jobs=N_JOBS,
                                     C=0.1,
                                     penalty='l2'
                                 ))], verbose=True)

rf = RandomForestClassifier(
    random_state=SEED,
    class_weight='balanced',
    n_jobs=N_JOBS,
    max_depth=20,
    n_estimators=2000
)

rfe = RFE(RandomForestClassifier(
    random_state=SEED,
    class_weight='balanced',
    n_estimators=2000,
    max_depth=20,
    n_jobs=N_JOBS
), verbose=10, n_features_to_select=500, step=0.1)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)

ada = AdaBoostClassifier(
    random_state=SEED,
    learning_rate=0.5,
    n_estimators=400
)

gb = Pipeline([('scaler', StandardScaler()),
               ['GB', GradientBoostingClassifier(
                   loss="deviance",
                   learning_rate=0.15,
                   min_samples_split=3,
                   min_samples_leaf=0.1,
                   max_depth=10,
                   max_features="sqrt",
                   criterion="mse",
                   subsample=0.8,
                   n_estimators=50
               )]])

algorithms = {
    'lg': lg_model,
    'lg_pca': lg_pca,
    'lg_selection': lg_feature_selection,
    'rf': rf,
    'rf_rfe': rfe,
    'knn': knn,
    'ada': ada,
    'gb': gb
}

if exists(results_path):
    results = pickle.load(open(results_path, 'rb'))
else:
    results = {}

for key, model in algorithms.items():
    print(f"Working on algorithm {key}")

    # if f"{key}_{path}" in results.keys():
    #     continue

    print(f"Working on the path: {path}")
    data = pickle.load(open(path, 'rb'))
    features_list = data['features']
    labels_list = data['labels']

    X = np.array(features_list)
    y = np.array(labels_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    model.fit(X_train, y_train)

    print('-----')
    y_pred_prob = model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    test_eer = EER(y_pred_prob, y_test)

    y_pred_prob = model.predict_proba(X_train)
    train_auc = roc_auc_score(y_train, y_pred_prob, multi_class='ovr')
    train_eer = EER(y_pred_prob, y_train)

    print(f"Train AUC:{train_auc}, Test AUC: {test_auc}")
    print(f"Train EER:{train_eer}, Test EER: {test_eer}")
    print('-----')

    results[f"{key}_{path}"] = {"train_auc": train_auc,
                                "test_auc": test_auc,
                                "train_eer": train_eer,
                                "test_eer": test_eer}

    pickle.dump(results, open(results_path, 'wb'))




