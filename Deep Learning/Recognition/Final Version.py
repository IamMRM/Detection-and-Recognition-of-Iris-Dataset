import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
#from torch.nn.functional import mse_loss
import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
#from torch.optim import lr_scheduler
import time
import cv2
from tqdm import tqdm
import copy
from torch.nn.modules.distance import PairwiseDistance


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
                                                num_train_triplets = 1120, num_valid_triplets = 1120,
                                                batch_size = 8,num_workers=0)


# In[5]:


import googlenet
model= googlenet.GoogLeNet().to(device)


# In[7]:


from torchsummary import summary
summary(model,input_size=(3,224,224))


# In[6]:


learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100
start = time.time()


# In[7]:


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

criterion=TripletLoss().to(device)


# In[8]:


def train(dataloader,epoch):
        model.train()
        losses=0
        for batch_idx, batch_sample in tqdm(enumerate(dataloader)):
            model.zero_grad()
            anchor_embedding=model(batch_sample["anc_img"].to(torch.float).to(device))
            positive_embedding=model(batch_sample["pos_img"].to(torch.float).to(device))
            negative_embedding=model(batch_sample["neg_img"].to(torch.float).to(device))
            
            loss=criterion(anchor_embedding,positive_embedding,negative_embedding)
            loss.backward()
            optimizer.step()
            losses+=loss.item()
        epoch_loss=losses/len(dataloader)
        return epoch_loss


# In[9]:


def evaluate(dataloader,epoch):    
        model.eval()
        val_loss_value=0
        with torch.no_grad():
            for batch_idx, batch_sample in tqdm(enumerate(dataloader)):
                anchor_embedding=model(batch_sample["anc_img"].to(torch.float).to(device))
                positive_embedding=model(batch_sample["pos_img"].to(torch.float).to(device))
                negative_embedding=model(batch_sample["neg_img"].to(torch.float).to(device))
                
                loss=criterion(anchor_embedding,positive_embedding,negative_embedding)
                val_loss_value+=loss.item()
            val_epoch_loss=val_loss_value/len(dataloader)
        return val_epoch_loss


# In[10]:


best_loss=999
losses=[]
training = []
validation = []

for epoch in range(num_epochs):
        train_loss=train(data_loaders['train'],epoch)
        print(f"TRAINING STEP: {epoch} LOSS: {train_loss}")
        training.append(train_loss)

        val_loss=evaluate(data_loaders['valid'],epoch)
        print(f"VALIDATING STEP: {epoch} LOSS: {val_loss}")
        validation.append(val_loss)
        
        if val_loss<best_loss:
            best_loss=val_loss
            print("Saving Best Model ",epoch)
            torch.save(model.state_dict(),os.path.join("","best_model.pth"))

print("DONE!!")