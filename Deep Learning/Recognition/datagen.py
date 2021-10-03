import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
import pandas as pd
import os
import cv2

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, csv_name, num_triplets, transform=None,train=True):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name, converters={'name': lambda x: str(x)})
        self.num_triplets = num_triplets
        self.transform = transform
        self.train=train
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets, train)

    @staticmethod
    def generate_triplets(df, num_triplets,train):#['10_R', '09_R', '01_L', 36, 146, '037', '147', 'bmp', 'bmp', 'bmp']
        def make_dictionary_for_face_class(df):
            '''face_classes = {'class0': [(class0_id0, extention),(..)]'''
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append((df.iloc[idx]['id'],df.iloc[idx]['mask']))
            return face_classes

        triplets = []
        classes = df['class'].unique()#sari classes ke naam
        face_classes = make_dictionary_for_face_class(df)#{0: [('01_L', 'bmp'), 
        for _ in range(num_triplets):
            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while pos_class == neg_class:#for random different class
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]
            ################FOR separating testing and training#############
            if train:
                temp_1=np.random.randint(0, len(face_classes[pos_class]))
                temp_2=np.random.randint(0, len(face_classes[pos_class]))
                temp_3=np.random.randint(0, len(face_classes[neg_class]))
                
                while temp_1 % 2!=0:
                    temp_1=np.random.randint(0, len(face_classes[pos_class]))
                while temp_2 % 2!=0 and temp_1 == temp_2:
                    temp_2 = np.random.randint(0, len(face_classes[pos_class]))
                while temp_3 % 2!=0:
                    temp_3=np.random.randint(0, len(face_classes[neg_class]))
                
                ianc = temp_1
                ipos = temp_2
                ineg = temp_3
                
            elif train==False:
                temp_1=np.random.randint(0, len(face_classes[pos_class]))
                temp_2=np.random.randint(0, len(face_classes[pos_class]))
                temp_3=np.random.randint(0, len(face_classes[neg_class]))
                
                while temp_1 % 2==0:
                    temp_1=np.random.randint(0, len(face_classes[pos_class]))
                while temp_2 % 2==0 and temp_1 == temp_2:
                    temp_2 = np.random.randint(0, len(face_classes[pos_class]))
                while temp_3 % 2==0:
                    temp_3=np.random.randint(0, len(face_classes[neg_class]))
                
                ianc = temp_1
                ipos = temp_2
                ineg = temp_3
            ###############################till here####################33
            anc_id = face_classes[pos_class][ianc][0]
            pos_id = face_classes[pos_class][ipos][0]
            neg_id = face_classes[neg_class][ineg][0]
            
            anc_mask=face_classes[pos_class][ianc][1]
            pos_mask=face_classes[pos_class][ipos][1]
            neg_mask= face_classes[neg_class][ineg][1]
            #print(anc_mask)
            
            triplets.append([anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_mask, pos_mask, neg_mask])
        return triplets

    def __getitem__(self, idx):
        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name, anc_mask, pos_mask, neg_mask = self.training_triplets[idx]
        fold_name=str(pos_name)
        
        
        if len(fold_name) == 2:
            anc_img = os.path.join(self.root_dir, '0'+fold_name, str(anc_id))
            pos_img = os.path.join(self.root_dir, '0'+fold_name, str(pos_id))
        elif len(fold_name) == 3:
            anc_img = os.path.join(self.root_dir, fold_name, str(anc_id))
            pos_img = os.path.join(self.root_dir, fold_name, str(pos_id))
        else:
            anc_img = os.path.join(self.root_dir, '00'+fold_name, str(anc_id))
            pos_img = os.path.join(self.root_dir, '00'+fold_name, str(pos_id))
        
        if len(str(neg_name))==2:
            neg_img = os.path.join(self.root_dir, '0'+str(neg_name), str(neg_id))
        elif len(str(neg_name))==3:
            neg_img = os.path.join(self.root_dir, str(neg_name), str(neg_id))
        else:
            neg_img = os.path.join(self.root_dir, '00'+str(neg_name), str(neg_id))
               
        anc_maskk = os.path.join((self.root_dir).replace("images","groundtruth"),str(anc_mask))
        pos_maskk=os.path.join((self.root_dir).replace("images","groundtruth"),str(pos_mask))
        neg_maskk = os.path.join((self.root_dir).replace("images","groundtruth"),str(neg_mask))
        
        """#TESTING
        print("ancr mask: "+str(anc_maskk))
        print("pos_mask : "+str(pos_maskk))
        print("neg_mask : "+str(neg_maskk))
        
        print(anc_img)
        print(pos_img)
        print(neg_img)"""
        anc_img = cv2.cvtColor(cv2.imread(anc_img), cv2.COLOR_BGR2RGB)
        pos_img = cv2.cvtColor(cv2.imread(pos_img), cv2.COLOR_BGR2RGB)
        neg_img = cv2.cvtColor(cv2.imread(neg_img), cv2.COLOR_BGR2RGB)
        
        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        anc_maskk = cv2.imread(anc_maskk, cv2.IMREAD_UNCHANGED)/ 255.0
        pos_maskk = cv2.imread(pos_maskk, cv2.IMREAD_UNCHANGED)/ 255.0
        neg_maskk = cv2.imread(neg_maskk, cv2.IMREAD_UNCHANGED)/255.0
        
        #print(np.unique(np.array(anc_maskk)))
        #print(anc_img.shape)
        
        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])
               
        """result1 = np.zeros(shape=(3,320,320))
        result2 = np.zeros(shape=(3,320,320))
        result3 = np.zeros(shape=(3,320,320))
        
        result1[:,:sample['anc_img'].shape[0],:sample['anc_img'].shape[1]] = np.moveaxis(sample['anc_img'],-1,0)*np.array(anc_maskk)
        result2[:,:sample['pos_img'].shape[0],:sample['pos_img'].shape[1]] = np.moveaxis(sample['pos_img'],-1,0)*np.array(pos_maskk)
        result3[:,:sample['neg_img'].shape[0],:sample['neg_img'].shape[1]] = np.moveaxis(sample['neg_img'],-1,0)*np.array(neg_maskk)
        
        sample['anc_img'] = result1
        sample['pos_img'] = result2
        sample['neg_img'] = result3"""
        
        sizee=224
        sample['anc_img'] = np.moveaxis((cv2.resize((np.moveaxis(np.moveaxis(sample['anc_img'],-1,0)*np.array(anc_maskk),0,-1)),dsize=(sizee,sizee), interpolation=cv2.INTER_AREA)),-1,0)
        sample['pos_img'] = np.moveaxis((cv2.resize((np.moveaxis(np.moveaxis(sample['pos_img'],-1,0)*np.array(pos_maskk),0,-1)),dsize=(sizee,sizee), interpolation=cv2.INTER_AREA)),-1,0)
        sample['neg_img'] = np.moveaxis((cv2.resize((np.moveaxis(np.moveaxis(sample['neg_img'],-1,0)*np.array(neg_maskk),0,-1)),dsize=(sizee,sizee), interpolation=cv2.INTER_AREA)),-1,0)
        
        return sample

    def __len__(self):
        return len(self.training_triplets)