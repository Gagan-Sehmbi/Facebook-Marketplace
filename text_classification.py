#%%
# IMPORT LIBRARIES

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

import transformers
from transformers import DistilBertTokenizer, DistilBertModel

from torch.utils.tensorboard import SummaryWriter

#%%
# IMPORT DATA

df = pd.read_csv('tabular_data_train.csv')

encoder = LabelEncoder()

df['label'] = encoder.fit_transform(df['Category'])
df['label'] = df['label'].astype('int64')
num_classes = max(df['label'])+1

# %%
# CREATE CUSTOM DATASET

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

class DataPrep(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.len = len(df)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data['Description'][index])
        embeddings = self.tokenizer(text,
                                    add_special_tokens = True,
                                    max_length = self.max_len,
                                    truncation = True,
                                    padding = 'max_length')
        
        return {
            'idx': torch.tensor(index),
            'ids': torch.tensor(embeddings['input_ids']),
            'mask': torch.tensor(embeddings['attention_mask']),
            'target': torch.tensor(self.data['label'][index])
        }
            
            
    def __len__(self):
            return self.len
                                    
                                    
df_train, df_val = train_test_split(df, test_size=0.25, shuffle=True)
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)

dataset_train = DataPrep(df_train, tokenizer, 256)
dataset_val = DataPrep(df_train, tokenizer, 256)

dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=True)


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
    path = "models/text_model.pth"
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
        #print(f'Epoch{epoch+1}: Validation Performance\n', cr)
        validation_score = f1_score(r, g, average= None)
        return validation_score

def fit(epochs, lr, model, train_loader, val_loader):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    
    best_score = 0.0

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
        #print(f'Epoch{epoch+1}: Training Performance\n', cr)
        training_score = f1_score(r, g, average=None)
        validation_score = evaluate(model, val_loader, epoch)

        print(f'Epoch{epoch+1}')
        print(f'best score: {best_score} - validation score: {np.mean(validation_score)}')

        if np.mean(validation_score) > best_score:
            saveModel()
            best_score = np.mean(validation_score)

    return training_score, validation_score


training_score, validation_score = fit(epochs=10,
                                       lr=1e-5, 
                                       model=model, 
                                       train_loader=dataloader_train,
                                       val_loader=dataloader_val)


# %%

