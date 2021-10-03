from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.io import imread
from skimage import img_as_float64


class IrisDataset(Dataset):

    def __init__(self, base_path, labels_df, transform=None):
        super(IrisDataset, self).__init__()

        self.base_path = base_path
        self.labels_df = labels_df
        self.transform = transform if transform is not None else self.get_basic_transform()

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_relative_path = row['images_path']
        target_relative_path = row['masks_path']
        class_id = row['classes']
        box = row[['x1', 'y1', 'x2', 'y2']].values

        image_path = join(self.base_path, image_relative_path)
        mask_path = join(self.base_path, target_relative_path)

        image, mask = self.read_images(image_path, mask_path)
        transformed = self.transform(image=image, mask=mask, bboxes=[box], class_id=class_id)
        t_image = transformed['image'].float()
        t_mask = transformed['mask'].unsqueeze(0).float()
        t_box = transformed['bboxes'][0]
        t_label = transformed['class_id']
        area = (t_box[3] - t_box[1]) * (t_box[2] - t_box[0])

        # data = {'masks': t_mask,
        #         'labels': torch.tensor([[t_label]], dtype=torch.int64),
        #         'boxes': torch.tensor([t_box], dtype=torch.float32),
        #         'image_id': torch.tensor([idx], dtype=torch.int64),
        #         'area': area}

        # weights = IrisDataset.get_weighted_loss(t_mask)
        return t_image, t_mask

    @staticmethod
    def get_weighted_loss(image):
        from skimage.morphology import erosion, square
        import matplotlib.pyplot as plt

        depth_level = 3
        temp_image = np.squeeze(image.numpy())
        weight_mask = np.zeros_like(temp_image) + 0.5
        for level in range(depth_level):
            eroded = erosion(temp_image, square(7))
            diff = temp_image - eroded
            weight_mask = np.where(diff != 0, 1 - level*0.1, weight_mask)

            temp_image -= diff

        # plt.imshow(weight_mask, cmap='gray')
        # plt.show()

        return weight_mask

    @staticmethod
    def Iris_collate_fn(batch):
        images_list = [item[0] for item in batch]
        targets_list = [item[1] for item in batch]
        return images_list, targets_list

    def __len__(self):
        return self.labels_df.shape[0]

    def get_basic_transform(self):
        transform = A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc',
                                                                       label_fields=['class_id']))
        return transform

    def read_images(self, image_path, target_path):
        # img = Image.open(image_path).convert('RGB')
        # target = Image.open(target_path)
        img = imread(image_path)
        target = imread(target_path)
        return img_as_float64(img), img_as_float64(target)


if __name__ == '__main__':
    import pandas as pd

    b = '/home/mahmoudk/iris/dataset'
    p = '/home/mahmoudk/iris/dataset/labels.csv'
    df = pd.read_csv(p)

    obj = IrisDataset(b, df)
    d = DataLoader(obj, batch_size=1, shuffle=True)

    import matplotlib.pyplot as plt
    shapes_list = []

    total_pixels_num = 0
    pos_pixels = 0
    neg_pixels = 0

    for img, target in d:
        print(img.shape, target.shape)
        # plt.imshow(img[0].permute((1, 2, 0)).numpy())
        # plt.show()
        # plt.imshow(target[0].permute((1, 2, 0)).numpy())
        # plt.show()
        # torch.max(img[0])
        total_pixels_num += np.prod(target.shape)
        pos_pixels += np.count_nonzero(target)
        neg_pixels += np.prod(target.shape) - np.count_nonzero(target)

        # shapes_list.append(img[0].shape)
        # shapes_list.append(img[1].shape)

    print(neg_pixels / pos_pixels)
