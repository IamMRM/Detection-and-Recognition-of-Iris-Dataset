import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib
from tqdm import tqdm

import googlenet
from sklearn.metrics import f1_score

# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


import datagen
def get_dataloader(train_root_dir,
                   train_csv_name,
                   num_train_triplets, num_valid_triplets,
                   batch_size, num_workers):

    face_dataset = {'train': datagen.TripletFaceDataset(root_dir=train_root_dir,csv_name=train_csv_name,
                                                num_triplets=num_train_triplets,transform=None,train=True),
                    'valid': datagen.TripletFaceDataset(root_dir=train_root_dir,csv_name=train_csv_name,
                                    num_triplets=num_valid_triplets,transform=None,train=False)}

    dataloaders = {x: torch.utils.data.DataLoader(face_dataset[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for x in ['train', 'valid']}
    
    data_size = {x: len(face_dataset[x]) for x in ['train', 'valid']}
    return dataloaders, data_size


# In[4]:


folder_dir = 'dataset/images/'
train_csv_name ='file.csv'
data_loaders, data_size = get_dataloader(train_root_dir = folder_dir,
                                                train_csv_name = train_csv_name,
                                                num_train_triplets =1120, num_valid_triplets =1120,
                                                batch_size = 1,num_workers=0)


# In[5]:


model = googlenet.GoogLeNet().cuda()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()


# In[6]:


def evaluate(dataloader,epoch):
        with torch.no_grad():
            for batch_idx, batch_sample in tqdm(enumerate(dataloader)):
                positive_embedding=model(batch_sample["pos_img"].to(torch.float).to(device))
                negative_embedding=model(batch_sample["neg_img"].to(torch.float).to(device))
                pos_class = torch.squeeze(batch_sample['pos_class'].to(torch.int64)).to(torch.float).cpu().numpy()
                neg_class = torch.squeeze(batch_sample['neg_class'].to(torch.int64)).to(torch.float).cpu().numpy()
                embeddings.append(torch.squeeze(positive_embedding).cpu().numpy())
                embeddings.append(torch.squeeze(negative_embedding).cpu().numpy())
                labels.append(pos_class)
                labels.append(neg_class)
        return epoch


# In[7]:


#for training
num_epochs=30
embeddings=[]
labels=[]
for epoch in range(num_epochs):
    epochs=evaluate(data_loaders['train'],epoch)

knn=KNeighborsClassifier(224)


# In[9]:


Ah = np.vstack(np.array(embeddings))
Bh = np.vstack(np.array(labels))
knn.fit(Ah,Bh)


# In[17]:


import pickle
knnPickle = open('knnpickle_file', 'wb')
pickle.dump(knn, knnPickle)
loaded_model = pickle.load(open('knnpickle_file', 'rb'))


def calc_scores(trueY, predY):
    return f1_score(trueY, predY, average='micro')

def eve(dataloader,epoch):
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_sample in tqdm(enumerate(dataloader)):
                positive_embedding=model(batch_sample["pos_img"].to(torch.float).to(device))
                pos_class = batch_sample['pos_class'].to(torch.int64).to(torch.float).cpu().numpy()
        return positive_embedding.cpu().numpy(), pos_class


# In[14]:


a=0.0
y_predList=[]
for i,X in enumerate(Ah):
    y_pred= knn.predict(np.expand_dims(X,axis=0))
    y_predList.append(y_pred)
print(calc_scores(Bh,y_predList))


# In[23]:


def evaluateV(dataloader,epoch):
        with torch.no_grad():
            for batch_idx, batch_sample in tqdm(enumerate(dataloader)):
                positive_embedding=model(batch_sample["pos_img"].to(torch.float).to(device))
                negative_embedding=model(batch_sample["neg_img"].to(torch.float).to(device))
                pos_class = torch.squeeze(batch_sample['pos_class'].to(torch.int64)).to(torch.float).cpu().numpy()
                neg_class = torch.squeeze(batch_sample['neg_class'].to(torch.int64)).to(torch.float).cpu().numpy()
                embeddingsV.append(torch.squeeze(positive_embedding).cpu().numpy())
                embeddingsV.append(torch.squeeze(negative_embedding).cpu().numpy())
                labelsV.append(pos_class)
                labelsV.append(neg_class)
        return epoch

#for testing
num_epochs=30
embeddingsV=[]
labelsV=[]
for epoch in range(num_epochs):
    epochs=evaluateV(data_loaders['valid'],epoch)
    print("Done with ",str(epochs))


# In[25]:

Av = np.vstack(np.array(embeddingsV))
Bv = np.vstack(np.array(labelsV))

# In[27]:

y_predListV=[]
for i,X in enumerate(Av):
    y_pred= knn.predict(np.expand_dims(X,axis=0))
    y_predListV.append(y_pred)
print(calc_scores(Bv,y_predListV))