#importing libraries
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import cv2
import copy
from sklearn.metrics import f1_score


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "modified_unet"
validation_loss = []
trains_loss = []


# In[3]:


class IrisDataset(Dataset):
    def __init__(self, train=True):
        self.directory_orig = 'D:\\Work\\MASTERS\\2nd Semester\\Machine and Deep Learning (ECTS 6)\\Project\\dataset\\images\\'
        self.directory_mask = 'D:\\Work\\MASTERS\\2nd Semester\\Machine and Deep Learning (ECTS 6)\\Project\\dataset\\groundtruth\\'

        self.input_images=[]
        for foldername in os.listdir(self.directory_orig):
            for index,filename in enumerate((os.listdir(self.directory_orig+foldername))):
                if train == True and index % 2 != 0:
                    if filename.endswith(".bmp"):
                        self.input_images.append(self.directory_orig+foldername+"//"+filename)
                elif train == False and index % 2 == 0:
                    if filename.endswith(".bmp"):
                        self.input_images.append(self.directory_orig+foldername+"//"+filename)
                
        self.target_masks=[]
        for index,foldername in enumerate(os.listdir(self.directory_mask)):
            if train == True and index % 2 != 0:
                if foldername.endswith(".tiff"):
                    self.target_masks.append(self.directory_mask+foldername)
            elif train == False and index % 2 == 0:
                if foldername.endswith(".tiff"):
                    self.target_masks.append(self.directory_mask+foldername)
        
        assert len(self.input_images) == len(self.target_masks)
    
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        image = torch.unsqueeze(torch.Tensor(cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY),dsize=(212,212), interpolation=cv2.INTER_AREA))/ 255.0,-1)
        mask = torch.unsqueeze(torch.Tensor(cv2.resize(cv2.imread(mask, cv2.IMREAD_UNCHANGED),dsize=(212,212), interpolation=cv2.INTER_AREA))/ 255.0,-1)
        mask[mask<.5]=0
        mask[mask>=.5]=1
        #print(mask.shape)
        return [image.permute(2, 0, 1), mask.permute(2, 0, 1)]


# In[4]:


train_set = IrisDataset(True)
valid_set = IrisDataset(False)
batch_size = 1
dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True), 
               'val': DataLoader(valid_set, batch_size=batch_size, shuffle=True)}
#print(len(train_set))
#print(len(valid_set))

if model_name == "baseline_model":
    import baseline_model
    model = baseline_model.L15()
    model = model.to(device)
elif model_name == "modified_unet":
    import modified_unet
    model = modified_unet.Unet()
    model = model.to(device)
else:
    print("************************Select an accurate model************************")


# In[7]:


# check keras-like model summary using torchsummary
from torchsummary import summary
summary(model, input_size=(1, 212, 212))


# In[6]:


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


# In[7]:


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = torch.logical_and(labels, outputs).sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = union = torch.logical_or(labels, outputs).sum((1, 2))         # Will be zzero if both are 0
    
    iou = torch.sum(intersection) / torch.sum(union)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


# In[10]:


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    pre=pred.detach().clone().cpu()
    pre[pre<.5]=0
    pre[pre>=.5]=1
    iou_score = iou_pytorch(pre,target.cpu())
    
    tar= target.detach().clone().cpu().numpy()
    tar=tar.reshape(batch_size*212*212*1)
    pre=pre.numpy().reshape(batch_size*212*212*1)
    f1__score = f1_score(tar,pre)
    
    
    metrics['f1 score'] += f1__score
    metrics['iou'] += iou_score.data.cpu().numpy()
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    if phase == "train":
        trains_loss.append(outputs)
    else:
        validation_loss.append(outputs)
    print("{}: {}".format(phase, ", ".join(outputs)))


# In[11]:


def train_model(model, optimizer, scheduler, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            metrics = defaultdict(float)
            epoch_samples = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)
            print_metrics(metrics, epoch_samples, phase)

            count = 0

            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                pred = 255.0 * np.array(model(inputs).permute(0, 2, 3, 1).to("cpu").detach())
                inputs = 255.0 * np.array(inputs.permute(0, 2, 3, 1).to("cpu").detach())
                labels = 255.0 * np.array(labels.permute(0, 2, 3, 1).to("cpu").detach())

                pred = np.hstack(pred)
                inputs = np.hstack(inputs)
                labels = np.hstack(labels)

                cv2.imwrite("pred_{}.jpg".format(epoch), pred)
                cv2.imwrite("labels_{}.jpg".format(epoch), labels)
                cv2.imwrite("inputs_{}.jpg".format(epoch), inputs)

                count += 1

                if count == 1:
                    break

            epoch_loss = metrics['loss'] / epoch_samples
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[12]:


num_epochs = 100
start = time.time()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
print(time.time() - start)
torch.save(model.state_dict(), r"model.pth")