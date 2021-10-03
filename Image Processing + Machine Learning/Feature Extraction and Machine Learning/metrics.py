import pickle

import numpy as np

# Decidability index calculation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


def decidability_index(features: np.ndarray, labels: np.ndarray):
    assert features.shape[0] == labels.shape[0]

    distro_mean = []
    distro_std = []
    for current_label in np.unique(labels):
        features_per_label = []
        for feature, label in zip(features, labels):
            if label == current_label:
                features_per_label.append(feature)
        distro_mean.append(np.mean(features_per_label, axis=0))
        distro_std.append(np.std(features_per_label, axis=0))

    distro_mean = np.array(distro_mean)
    distro_std = np.array(distro_std)

    average_std = np.sqrt((distro_std**2 + np.expand_dims(distro_std**2, axis=1)) * 0.5)
    mean_diff = np.abs(distro_mean - np.expand_dims(distro_mean, axis=1))
    di = np.divide(mean_diff, average_std, out=np.zeros_like(mean_diff), where=average_std != 0)
    print(np.mean(di))
    return np.mean(di)


def EER(predictions: np.ndarray, labels: np.ndarray):

    assert predictions.shape[0] == labels.shape[0]

    eer_per_class = []
    for class_id in np.unique(labels):
        fpr, tpr, threshold = roc_curve(y_score=predictions[:, class_id-1], y_true=labels, pos_label=class_id)
        fnr = 1 - tpr
        eer1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_per_class.append(eer1)

    print(np.mean(eer_per_class))
    return np.mean(eer_per_class)


if __name__ == '__main__':
    base_path = "/home/mahmoudk/iris/dataset/HOG/hog_16_(6, 10).pkl"

    d = pickle.load(open(base_path, 'rb'))
    features_list = d['features']
    labels_list = d['labels']

    X = np.array(features_list) # n x feature vector size
    y = np.array(labels_list) # n labels

    decidability_index(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    # print("Training LR")
    model_logistic_regression = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        solver="liblinear",
        C=50,
        penalty="l2",
        verbose=10,
        n_jobs=10
    )

    model_logistic_regression.fit(X_train, y_train)
    prob = model_logistic_regression.predict_proba(X_test)
    # prob = model_logistic_regression.predict_proba(X_train)

    EER(prob, y_test)




