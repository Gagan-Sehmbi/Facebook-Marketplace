#%%
# IMPORT LIBRARIES

import random

import cv2
from PIL import Image

import glob
import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

import transformers
from transformers import DistilBertTokenizer, DistilBertModel

from torch.utils.tensorboard import SummaryWriter


# %%
# IMPORT IMAGE DATA
# CREATE TRAIN AND VAL SETS

train_data_path = 'images/train' 

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))
    
train_image_paths = [item for sublist in train_image_paths for item in sublist]
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

#split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.75*len(train_image_paths))], train_image_paths[int(0.75*len(train_image_paths)):] 

print("\nTrain size: {}\nValid size: {}".format(len(train_image_paths), len(valid_image_paths)))
#%%
# IMPORT TEXT DATA

df = pd.read_csv('tabular_data_train.csv')

encoder = LabelEncoder()

df['label'] = encoder.fit_transform(df['Category'])
df['label'] = df['label'].astype('int64')
num_classes = max(df['label'])+1

train_ids = [item.split('/')[-1][:-4] for item in train_image_paths]
val_ids = [item.split('/')[-1][:-4] for item in valid_image_paths]

df_train = df.loc[df['id'].isin(train_ids)]
df_val = df.loc[df['id'].isin(val_ids)]
# %%
# CREATE IMAGE TRANSFORM FUNCTION

def transform(img, final_dim=128):
    img_size = img.size
    max_dim = max(img.size)
    sf = final_dim/max_dim

    new_img_size = (int(img_size[0]*sf), int(img_size[1]*sf))
    new_img = img.resize(new_img_size)

    final_img = Image.new(mode='RGB', size=(final_dim, final_dim))
    final_img.paste(new_img, ((final_dim-new_img_size[0])//2, (final_dim-new_img_size[1])//2))

    output = transforms.ToTensor()(final_img)
    return output

# %%
# CREATE CUSTOM DATASET

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

class MMFBMDataset(Dataset):
    def __init__(self, image_paths, df, tokenizer, max_len):
        self.image_paths = image_paths
        self.transform = transform

        self.data = df
        self.len = len(df)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        image_filepath = self.image_paths[index]
        image = Image.open(image_filepath)

        if self.transform is not None:
            image = self.transform(img=image)

        text = self.data.loc[self.data['id'] == image_filepath.split('/')[-1][:-4], 'Description'].values[0]
        embeddings = self.tokenizer(text,
                                    add_special_tokens = True,
                                    max_length = self.max_len,
                                    truncation = True,
                                    padding = 'max_length')
        
        label = self.data.loc[self.data['id'] == image_filepath.split('/')[-1][:-4], 'label'].values[0]

        return {
            'idx': torch.tensor(index),
            'image': image,
            'ids': torch.tensor(embeddings['input_ids']),
            'mask': torch.tensor(embeddings['attention_mask']),
            'target': label
        }
            
            
    def __len__(self):
            return self.len
                                
dataset_train = MMFBMDataset(train_image_paths, df_train, tokenizer, 256)
dataset_val = MMFBMDataset(valid_image_paths, df_val, tokenizer, 256)

dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=True)

batch = next(iter(dataloader_train))
print(f"Image Tensor: {batch['image'].size()}")
print(f"Token Tensor: {batch['ids'].size()}")
print(f"Attention Mask Tensor: {batch['mask'].size()}")
print(f"Label Tensor: {batch['target'].size()}")

#%%
# MODEL

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class FBMClassifier(torch.nn.Module):
    def __init__(self):
        super(FBMClassifier, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.l2 = torch.nn.Linear(768,768)
        self.l3 = torch.nn.Dropout(0.3)
        self.l4 = torch.nn.Linear(768, num_classes)
        
    def forward(self, ids, mask):
        out = self.l1(ids, mask)
        out = out[0][:,0]
        out = self.l2(out)
        out = torch.nn.ReLU()(out)
        out = self.l3(out)
        out = self.l4(out)
        
        return out
    
    def step(self, batch):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        target = batch['target'].to(device)
        
        output = self(ids, mask)
        
        loss = loss_func(output, target)
        
        return output, target, loss


model = FBMClassifier()
model.to(device)
loss_func = torch.nn.CrossEntropyLoss()
opt_func = torch.optim.Adam


def saveModel():
    path = "./text_model.pth"
    torch.save(model.state_dict(), path)

def evaluate(model, val_loader, epoch):
    guess = []
    real = []
    with torch.no_grad():
        for batch in val_loader:
            output, target, loss = model.step(batch)
            _, pred = torch.max(output.data, dim=1)
            guess.append(pred.cpu().detach().tolist())
            real.append(target.cpu().detach().tolist())
        r = [j for i in real for j in i]
        g = [j for i in guess for j in i]
        cr = classification_report(r, g)
        print(f'Epoch{epoch+1}: Validation Performance\n', cr)
        validation_score = f1_score(r, g, average=None)
        return validation_score

def fit(epochs, lr, model, train_loader, val_loader):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    
    best_score = 0

    writer = SummaryWriter()

    n_iter = 0

    for epoch in range(epochs):
        guess = []
        real = []
        for batch in train_loader:
            output, target, loss = model.step(batch)
            _, pred = torch.max(output.data, dim=1)
            guess.append(pred.cpu().detach().tolist())
            real.append(target.cpu().detach().tolist())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('loss', loss.item(), n_iter)
            n_iter += 1

        r = [j for i in real for j in i]
        g = [j for i in guess for j in i]
        cr = classification_report(r, g)
        print(f'Epoch{epoch+1}: Training Performance\n', cr)
        training_score = f1_score(r, g, average=None)
        
        validation_score = evaluate(model, val_loader, epoch)

        if validation_score > best_score:
            saveModel()
            best_score = validation_score

    return training_score, validation_score


training_score, validation_score = fit(epochs=5,
                                       lr=1e-5, 
                                       model=model, 
                                       train_loader=dataloader_train,
                                       val_loader=dataloader_val)


# %%
