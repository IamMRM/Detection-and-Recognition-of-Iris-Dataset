from os.path import join, basename
from os import makedirs, listdir
import pickle

import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import integral_image
from skimage.feature import multiblock_lbp, hog

# base_path = "D:\\datasets\\iris\\dataset\\norms"
# labels_path = 'D:\\datasets\\iris\\dataset\\labels.csv'
# lbp_output_path = "D:\\datasets\\iris\\dataset\\LBP"

base_path = "/home/mahmoudk/iris/dataset/norms"
labels_path = '/home/mahmoudk/iris/dataset/labels.csv'
hog_output_path = "/home/mahmoudk/iris/dataset/HOG"

labels_df = pd.read_csv(labels_path)
# images_full_paths = [join(base_path, basename(p)) for p in labels_df['images_path'].values]
images_full_paths = [join(base_path, p) for p in sorted(listdir(base_path))]


def decimalize(y):
    z = np.array([], dtype=np.int64)
    for x in y:
        z = np.append(z, int(''.join(map(lambda x: str(int(x)), x)), 2))
    return z


## MB LBP Features

makedirs(hog_output_path, exist_ok=True)

orientation_list = [4, 8, 16]
block_size_list = [(6, 10)]

for orientation in orientation_list:
    for block_size in block_size_list:
        classes_list = []
        features_list = []

        for image_path in tqdm(images_full_paths, desc=f'HOG Feature extractio'):
            img = rgb2gray(imread(image_path))
            feature_vector = hog(img, orientations=orientation,
                                 pixels_per_cell=(block_size[0], block_size[1]),
                                 cells_per_block=(3, 3),
                                 visualize=False,
                                 block_norm='L2-Hys',
                                 feature_vector=True)

            class_id = int(basename(image_path).split('_')[0])
            classes_list.append(class_id)
            features_list.append(feature_vector)

        print(feature_vector.shape)
        d = {'features': features_list, 'labels': classes_list}
        pickle.dump(d, open(join(hog_output_path, f'hog_{orientation}_{block_size}.pkl'), 'wb'))
