from os.path import join, basename
from os import makedirs, listdir
import pickle

import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern


# base_path = "D:\\datasets\\iris\\dataset\\norms"
# labels_path = 'D:\\datasets\\iris\\dataset\\labels.csv'
# lbp_output_path = "D:\\datasets\\iris\\dataset\\LBP"

base_path = "/home/mahmoudk/iris/dataset/norms"
labels_path = '/home/mahmoudk/iris/dataset/labels.csv'
lbp_output_path = "/home/mahmoudk/iris/dataset/LBP"

labels_df = pd.read_csv(labels_path)
# images_full_paths = [join(base_path, basename(p)) for p in labels_df['images_path'].values]
images_full_paths = [join(base_path, p) for p in sorted(listdir(base_path))]


def decimalize(y):
    z = np.array([], dtype=np.int64)
    for x in y:
        z = np.append(z,int(''.join(map(lambda x: str(int(x)), x)), 2))
    return z

## LBP Features
radius_list = [1, 3, 5]

for radius in radius_list:
    n_points = 8 * radius
    dir_path = join(lbp_output_path, f'r_{radius}')
    makedirs(dir_path, exist_ok=True)
    classes_list = []
    features_list = []

    for image_path in tqdm(images_full_paths, desc=f'LBP Feature extractionn for R: {radius}'):
        img = rgb2gray(imread(image_path))
        lbp_image = local_binary_pattern(image=img, P=n_points, R=radius)
        output_path = join(dir_path, basename(image_path))
        # cv2.imwrite(filename=output_path, img=lbp_image.astype(np.uint64))

        class_id = int(basename(image_path).split('_')[0])
        classes_list.append(class_id)
        hist, _ = np.histogram(lbp_image.ravel(), bins=256)
        hist = hist / np.sum(hist)
        features_list.append(hist)

    d = {'features': features_list, 'labels': classes_list}
    pickle.dump(d, open(join(lbp_output_path, f'r_{radius}.pkl'), 'wb'))

concat_features = []
classes_list = []
for image_path in tqdm(images_full_paths, desc=f'LBP Concat Feature extraction'):
    features_list = []
    class_id = int(basename(image_path).split('_')[0])
    classes_list.append(class_id)
    for radius in radius_list:
        n_points = 8 * radius
        img = rgb2gray(imread(image_path))
        lbp_image = local_binary_pattern(image=img, P=n_points, R=radius)

        hist, _ = np.histogram(lbp_image.ravel(), bins=256)
        hist = hist / np.sum(hist)

        features_list.append(hist)

    concat_features.append(np.array(features_list).ravel())
    d = {'features': concat_features, 'labels': classes_list}
    pickle.dump(d, open(join(lbp_output_path, f'concat_lbp.pkl'), 'wb'))

