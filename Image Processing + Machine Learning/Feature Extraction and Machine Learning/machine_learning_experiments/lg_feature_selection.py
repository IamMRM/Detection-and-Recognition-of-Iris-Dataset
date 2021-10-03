import pickle
from os.path import join, basename
from os import makedirs, listdir

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler

# base_path = "D:\\datasets\\iris\\dataset\\LBP\\r_3"
# base_path = "/home/mahmoudk/iris/dataset/LBP/r_5"
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

# X = np.array(features_list)
# y = np.array(classes_list)

# base_path = "/home/mahmoudk/iris/dataset/MB_LBP/h_3_w_5.pkl"
base_path = "/home/mahmoudk/iris/dataset/HOG/hog.pkl"
d = pickle.load(open(base_path, 'rb'))
features_list = d['features']
labels_list = d['labels']

X = np.array(features_list)
y = np.array(labels_list)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
parameters = {
    # "feature_selection__k": [500, 1000, 1500],
    "pca__n_components": [100, 500, 1000, 1400],
    "lg__C": [50],
    "lg__penalty": ["l2"]
}
SEED = 123
model_logistic_regression = LogisticRegression(
    random_state=SEED,
    class_weight="balanced",
    solver="liblinear",
    max_iter=300
)

pipe = Pipeline([('scalar', StandardScaler()),
                 # ('feature_selection', SelectKBest(mutual_info_classif)),
                 ('pca', PCA()),
                 ('lg', model_logistic_regression)])


gv_model_logistic_regression = GridSearchCV(
    pipe,
    parameters,
    cv=StratifiedKFold(n_splits=5),
    scoring= 'roc_auc_ovr',
    verbose = 10,
    n_jobs=30
)

gv_model_logistic_regression.fit(X_train_scaled, y_train)

print('-----')
print(f'Best parameters {gv_model_logistic_regression.best_params_}')
print(
    f'Mean cross-validated accuracy score of the best_estimator: ' +
    f'{gv_model_logistic_regression.best_score_:.3f}'
)

print('-----')

y_pred = gv_model_logistic_regression.predict(X_test_scaled)
print(f"F1 score: {f1_score(y_true=y_test, y_pred=y_pred, average='macro')}")

y_pred = gv_model_logistic_regression.predict_proba(X_test_scaled)
print(roc_auc_score(y_test, y_pred, multi_class='ovr'))

