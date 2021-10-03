from os.path import join, basename
from os import makedirs, listdir

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler

# base_path = "D:\\datasets\\iris\\dataset\\LBP\\r_3"
base_path = "/home/mahmoudk/iris/dataset/LBP/r_1"

print(f"Running on {base_path}")

images_paths = [join(base_path, f) for f in sorted(listdir(base_path))]

features_list = []
classes_list = []

for image_path in images_paths:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features_list.append(img.reshape(-1))
    classes_list.append(int(basename(image_path).split('_')[0]))

X = np.array(features_list)
y = np.array(classes_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

parameters =    {
    'n_neighbors': [1, 3, 5, 7]
}

model_KNN =  KNeighborsClassifier()

model_KNN = GridSearchCV(
    model_KNN,
    parameters,
    cv=StratifiedKFold(n_splits=5),
    scoring='roc_auc_ovr',
    verbose = 10,
    n_jobs = 30,
)

model_KNN.fit(X_train, y_train)

print('-----')
print(f'Best parameters {model_KNN.best_params_}')
print(
    f'Mean cross-validated accuracy score of the best_estimator: ' +
    f'{model_KNN.best_score_:.3f}'
)
print('-----')

y_pred = model_KNN.best_estimator_.predict(X_test)
print(f1_score(y_test,y_pred ,average = "macro"))

y_pred = model_KNN.best_estimator_.predict_proba( X_test)
auc = roc_auc_score(y_true=y_test, y_score=y_pred, multi_class='ovr')
print(f"KNN test auc:{auc}")

