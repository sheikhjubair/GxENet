#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
from scipy.stats import pearsonr
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# In[ ]:


def evaluate_step(dataloader, model, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    val_loss = 0
    predicted = []
    true = []
    true_original_space = []
    
    eval_loss = 0
    for locations, trials, data, target, target_raw in dataloader:
        data = data.to(device)
        data = data.float()
        target = target.to(device)
        target = target.float()
        
        outputs = model(data)
        outputs= outputs.view(-1,)
        
        loss = criterion(target, outputs)
        
        eval_loss += loss.item()
        
        predicted += outputs.detach().cpu().numpy().tolist()
        true += target.detach().cpu().numpy().tolist()

    eval_loss = eval_loss / len(dataloader)    
    predicted = np.array(predicted)
    true = np.array(true)
    
    return eval_loss, true, predicted


# In[ ]:


def eval(data, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.no_grad():
        model = model.to(device)
        model.eval()

        data = data.to(device)
        data = data.float()

        outputs, representation = model(data)
        outputs= outputs.view(-1,)
    
    return outputs, representation


# In[ ]:


def evaluate_step_by_avg(dataloader, model, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    val_loss = 0
    predicted = []
    true = []
    true_original_space = []
    
    eval_loss = 0
    for data, target in dataloader:
        data = data.to(device)
        data = data.float()
        target = target.to(device)
        target = target.float()
        
        outputs, _ = model(data)
        outputs= outputs.view(-1,)
        
        loss = criterion(target, outputs)
        
        eval_loss += loss.item()
        
        predicted += outputs.detach().cpu().numpy().tolist()
        true += target.detach().cpu().numpy().tolist()

    eval_loss = eval_loss / len(dataloader)    
    predicted = np.array(predicted)
    true = np.array(true)
    
    return eval_loss, true, predicted

