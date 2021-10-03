from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.io import imread
from skimage import img_as_float64


class SiameseIrisTrainDataset(Dataset):

    def __init__(self, base_path, labels_df, transform=None):
        super(SiameseIrisTrainDataset, self).__init__()

        self.base_path = base_path
        self.labels_df = labels_df
        self.transform = transform if transform is not None else self.get_basic_transform()

    def __getitem__(self, idx):
        image1_row = self.labels_df.iloc[idx]
        image1_class_id = image1_row['classes']

        same_class = np.random.rand() > 0.5

        if same_class:
            image2_row = self.labels_df[self.labels_df['classes'] == image1_class_id].sample()
            label = 1
        else:
            image2_row = self.labels_df[self.labels_df['classes'] != image1_class_id].sample()
            label = 0

        image1_relative_path = image1_row['images_path']
        image2_relative_path = image2_row['images_path'].item()

        image1_path = join(self.base_path, image1_relative_path)
        image2_path = join(self.base_path, image2_relative_path)

        image1, image2 = self.read_images(image1_path, image2_path)
        transformed = self.transform(image=image1, image1=image2)
        t_image1 = transformed['image'].float()
        t_image2 = transformed['image1'].float()

        return t_image1, t_image2, torch.tensor([label], dtype=torch.float)

    def __len__(self):
        return self.labels_df.shape[0]

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

    obj = SiameseIrisTrainDataset(b, df)
    d = DataLoader(obj, batch_size=2, shuffle=True)

    import matplotlib.pyplot as plt
    images_mean = torch.zeros(3)
    images_std = torch.zeros(3)
    from tqdm import tqdm

    progress_bar = tqdm(total=len(d),
                        desc=f'Batches')

    for img1, img2, label in d:
        images_mean += img1.mean(dim=[0, 2, 3]) + img2.mean(dim=[0, 2, 3])
        images_std += img1.std(dim=[0, 2, 3]) + img2.std(dim=[0, 2, 3])
        progress_bar.update(1)
    progress_bar.close()
    images_mean /= len(d)
    images_std /= len(d)

    print(f'Mean: {images_mean}, STD: {images_std}')