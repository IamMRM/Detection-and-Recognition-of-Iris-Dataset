from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.io import imread
from skimage import img_as_float64


class SiameseIrisTestDataset(Dataset):

    def __init__(self, base_path, labels_df, transform=None, is_class=False):
        super(SiameseIrisTestDataset, self).__init__()

        self.base_path = base_path
        self.labels_df = labels_df
        self.transform = transform if transform is not None else self.get_basic_transform()
        self.is_class = is_class

    def __getitem__(self, idx):
        unique_classes = self.labels_df['classes'].unique()
        chosen_class = unique_classes[idx]

        images_df = self.labels_df[self.labels_df['classes'] == chosen_class].sample(n=2)

        image1_relative_path = images_df['images_path'].iloc[0]
        image2_relative_path = images_df['images_path'].iloc[1]

        image1_path = join(self.base_path, image1_relative_path)
        image2_path = join(self.base_path, image2_relative_path)

        image1, image2 = self.read_images(image1_path, image2_path)
        transformed = self.transform(image=image1, image1=image2)
        t_image1 = transformed['image'].float()
        t_image2 = transformed['image1'].float()

        return t_image1, t_image2, torch.ones(1), torch.tensor([chosen_class], dtype=torch.float)


    def __len__(self):
        return self.labels_df['classes'].unique().shape[0]

    def get_basic_transform(self):
        transform = A.Compose([A.Normalize(mean=(0.9388, 0.9388, 0.9388), std=(0.5086, 0.5086, 0.5086)),
                               ToTensorV2()], additional_targets={'image1': 'image'})
        return transform

    def read_images(self, image_path, target_path):
        img = imread(image_path)
        target = imread(target_path)
        return img_as_float64(img), img_as_float64(target)


if __name__ == '__main__':
    import pandas as pd

    b = '/home/mahmoudk/iris/dataset'
    p = '/home/mahmoudk/iris/dataset/labels.csv'
    df = pd.read_csv(p)

    obj = SiameseIrisTestDataset(b, df)
    d = DataLoader(obj, batch_size=2, shuffle=True)

    import matplotlib.pyplot as plt

    for img1, img2, label, class_id in d:
        print(img1.shape, label.shape, label, class_id.shape, class_id)
        break
