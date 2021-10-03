from os.path import join, basename
from os import makedirs, listdir
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

# base_path = "D:\\datasets\\iris\\dataset\\LBP\\r_3"
# base_path = "/home/mahmoudk/iris/dataset/LBP/r_1"
#
# print(f"Running on {base_path}")
#
# images_paths = [join(base_path, f) for f in sorted(listdir(base_path))]
#
# features_list = []
# classes_list = []
#
# for image_path in images_paths:
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     features_list.append(img.reshape(-1))
#     classes_list.append(int(basename(image_path).split('_')[0]))
#
# X = np.array(features_list)
# y = np.array(classes_list)

base_path = "/home/mahmoudk/iris/dataset/MB_LBP/h_3_w_5.pkl"
d = pickle.load(open(base_path, 'rb'))
features_list = d['features']
labels_list = d['labels']

X = np.array(features_list)
y = np.array(labels_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.15, 0.7],
#     "min_samples_split": np.linspace(0.1, 0.5, 3),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 3),
#     "max_depth":[5, 10],
#     "max_features":["sqrt"],
#     "criterion": ["mae","mse"],
#     "subsample":[ 0.8, 0.95, 1.0],
#     "n_estimators":[10, 50]
# }

# model_GB =  GradientBoostingClassifier()
# model_GB = GridSearchCV(
#     model_GB,
#     parameters,
#     cv=StratifiedKFold(n_splits=5),
#     scoring='roc_auc_ovr',
#     verbose = 10,
#     n_jobs=30
# )


pipe = Pipeline([('scaler', StandardScaler()),
                 # ('feature_selection', SelectKBest(mutual_info_classif)),
                 ('GB', GradientBoostingClassifier())])

parameters = {
    # "feature_selection__k": [100, 300],
    "GB__loss":["deviance"],
    "GB__learning_rate": [0.15],
    "GB__min_samples_split": [3],
    "GB__min_samples_leaf": [0.1],
    "GB__max_depth":[10],
    "GB__max_features":["sqrt"],
    "GB__criterion": ["mse"],
    "GB__subsample":[0.8],
    "GB__n_estimators":[50]
}

gv_GB = GridSearchCV(
    pipe,
    parameters,
    cv=StratifiedKFold(n_splits=5),
    scoring='roc_auc_ovr',
    verbose = 10,
    n_jobs=15
)

gv_GB.fit(X_train, y_train)

print('-----')
print(f'Best parameters {gv_GB.best_params_}')
print(
    f'Mean cross-validated accuracy score of the best_estimator: ' +
    f'{gv_GB.best_score_:.3f}'
)
print('-----')

print(roc_auc_score(y_test, gv_GB.best_estimator_.predict_proba(X_test), multi_class='ovr'))
