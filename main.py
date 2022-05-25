# %%
# IMPORT LIBRARIES

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import cv2
from PIL import Image

import glob
import time

# %%
# CREATE TRAIN, VAL, TEST SETS

train_data_path = 'images/train' 
test_data_path = 'images/test'

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('/')[-1])
    train_image_paths.append(glob.glob(data_path + '/*'))
    
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])


#split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.75*len(train_image_paths))], train_image_paths[int(0.75*len(train_image_paths)):] 


test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))

print("\nTrain size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

# %% 
# CREATE INDEX_TO_CLASS AND CLASS_TO_INDEX

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
idx_to_class

# %%
# CREATE TRANSFORM FUNCTION

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
# CREATE DATASET

class FBMDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        
        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(img=image)
        
        return image, label

train_dataset = FBMDataset(train_image_paths,transform=transform)
valid_dataset = FBMDataset(valid_image_paths,transform=transform)
test_dataset = FBMDataset(test_image_paths,transform=transform)

print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[49][1])

# %%
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=64, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False
)

print(next(iter(train_loader))[0].shape)
print(next(iter(train_loader))[1].shape)
# %%
# TRAIN MODEL

