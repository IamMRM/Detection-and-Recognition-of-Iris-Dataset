from os.path import join, basename
from os import makedirs, listdir

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler

# base_path = "D:\\datasets\\iris\\dataset\\LBP\\r_3"
base_path = "/home/mahmoudk/iris/dataset/LBP/r_5"

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

model_random_forest = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_estimators=1000,
    max_depth=10
)

parameters = {
    "n_features_to_select": [1000],
    "step": [0.1],
}

selector = RFE(model_random_forest, verbose=10)

selector = GridSearchCV(
    selector,
    parameters,
    cv=StratifiedKFold(n_splits=2),
    scoring='roc_auc_ovr',
    verbose = 10,
    n_jobs=25
)

selector.fit(X_train, y_train)
print('-----')
print(f'Best parameters {selector.best_params_}')
print(
    f'Mean cross-validated accuracy score of the best_estimator: '+ \
    f'{selector.best_score_:.3f}'
)
print('-----')
y_pred = selector.best_estimator_.predict_proba( X_test)
auc = roc_auc_score(y_true=y_test, y_score=y_pred, multi_class='ovr')
print(f"RF tst auc:{auc}")


