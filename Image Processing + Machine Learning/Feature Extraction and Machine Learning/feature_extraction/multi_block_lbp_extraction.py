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
from skimage.feature import multiblock_lbp

# base_path = "D:\\datasets\\iris\\dataset\\norms"
# labels_path = 'D:\\datasets\\iris\\dataset\\labels.csv'
# lbp_output_path = "D:\\datasets\\iris\\dataset\\LBP"

base_path = "/home/mahmoudk/iris/dataset/norms"
labels_path = '/home/mahmoudk/iris/dataset/labels.csv'
lbp_output_path = "/home/mahmoudk/iris/dataset/MB_LBP"

labels_df = pd.read_csv(labels_path)
# images_full_paths = [join(base_path, basename(p)) for p in labels_df['images_path'].values]
images_full_paths = [join(base_path, p) for p in sorted(listdir(base_path))]


def decimalize(y):
    z = np.array([], dtype=np.int64)
    for x in y:
        z = np.append(z, int(''.join(map(lambda x: str(int(x)), x)), 2))
    return z


## MB LBP Features
shape_list = [(2, 10), (3, 5)]
for h, w in shape_list:
    dir_path = join(lbp_output_path, f'h_{h}_w_{w}')
    makedirs(dir_path, exist_ok=True)
    classes_list = []
    features_list = []

    for image_path in tqdm(images_full_paths, desc=f'MB-LBP Feature extraction for H: {h}, W: {w}'):
        img = rgb2gray(imread(image_path))
        int_img = integral_image(img)
        mb_vector = []
        block_width = 3 * w
        block_height = 3 * h
        for r in range(0, img.shape[0], block_height):
            for c in range(0, img.shape[1], block_width):
                lbp_feature = multiblock_lbp(int_img, r=r, c=c, width=w, height=h)
                mb_vector.append(lbp_feature)

        class_id = int(basename(image_path).split('_')[0])
        classes_list.append(class_id)
        features_list.append(mb_vector)

    d = {'features': features_list, 'labels': classes_list}
    pickle.dump(d, open(join(lbp_output_path, f'h_{h}_w_{w}.pkl'), 'wb'))
