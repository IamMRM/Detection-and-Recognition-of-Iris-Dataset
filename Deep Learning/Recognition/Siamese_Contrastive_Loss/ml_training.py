import pickle
from os.path import join

import torch
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm, trange
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage import img_as_float64
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt

from siamese_net import SiameseNet
from utils import save_model, load_model, save_pickle_file, load_pickle_file


model_path = '/home/mahmoudk/iris/save_dir/Siamese_train_without_sigmoid_V3/models/model_59.pth'
labels_path = '/home/mahmoudk/iris/dataset/labels.csv'
base_path = '/home/mahmoudk/iris/dataset'
knn_path = '/home/mahmoudk/iris/save_dir/Siamese_train_V1/knn.pth'
k = 2

def visualize_data(X, y):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    fig, axes = plt.subplots(1)
    fig.suptitle("Vector Representation in 2D Space")
    axes.set_xlabel('Dimension 1')
    axes.set_ylabel('Dimension 2')

    for label_id in np.unique(y):
        plt.scatter(x=X_embedded[y == label_id, 0], y=X_embedded[y == label_id, 1], label=label_id)

    plt.show()


model = SiameseNet()
model.eval()
load_model(model, model_path)
model = model.to('cuda:0')

labels_df = pd.read_csv(labels_path)
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=41, shuffle=True,
                                     stratify=labels_df['classes'].values)

# train_df = labels_df[labels_df['classes'] <= 112]
# test_df = labels_df[labels_df['classes'] > 112]

transform = A.Compose([A.Normalize(mean=(0.9388, 0.9388, 0.9388), std=(0.5086, 0.5086, 0.5086)),
                       ToTensorV2()])


# Embeddings Generation loop
progress_bar = tqdm(total=len(train_df), desc=f'Embeddings Calculated')
embeddings = []
classes = []
for i in range(train_df.shape[0]):
    image_row = train_df.iloc[i]
    image_class_id = image_row['classes']
    image_relative_path = image_row['images_path']
    image_path = join(base_path, image_relative_path)
    img = imread(image_path)

    t_image = transform(image=img)['image'].float()
    t_image = torch.unsqueeze(t_image, dim=0).to('cuda:0')

    embedding = model(t_image)
    embeddings.append(embedding.detach().to('cpu').numpy().squeeze())
    classes.append(image_class_id)

    progress_bar.update(1)

progress_bar.close()

X = np.array(embeddings)
y = np.array(classes)

visualize_data(X, y)

knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
knn_model.fit(X, y)
pickle.dump(knn_model, open(knn_path, 'wb'))

# Validation loop
# Embeddings Generation loop
progress_bar = tqdm(total=len(test_df), desc=f'Test predicted')
classes = []
preds = []
for i in range(test_df.shape[0]):
    image_row = test_df.iloc[i]
    image_class_id = image_row['classes']
    image_relative_path = image_row['images_path']
    image_path = join(base_path, image_relative_path)
    img = imread(image_path)

    t_image = transform(image=img)['image'].float()
    t_image = torch.unsqueeze(t_image, dim=0).to('cuda:0')

    embedding = model(t_image).detach().to('cpu').numpy().squeeze()
    classes.append(image_class_id)
    preds.append(knn_model.predict(embedding.reshape((1, -1))))

    progress_bar.update(1)

progress_bar.close()

preds = np.array(preds)
y_test = np.array(classes)

f1 = f1_score(y_pred=preds, y_true=y_test, average='micro')
print(f"F1 is {f1}")

