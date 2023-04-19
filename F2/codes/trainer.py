#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../codes')
import config


# In[2]:


import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from loss import mse_pcc_loss, calculate_perf_measure, calculate_loc_spec_pcc
from evaluator import evaluate_step, evaluate_step_by_avg
from IPython.display import clear_output


# In[3]:


def train_step(dataloader, model, criterion, optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    
    tr_running_loss = 0
    predicted = []
    true = []
 
    for locations, trials, data, target, target_raw in dataloader:
    
        data = data.to(device)
        data = data.float()
        target = target.to(device)
        target= target.float()
        
        optimizer.zero_grad()
        
        outputs = model(data)
        outputs= outputs.view(-1,)
        
        loss = criterion(target, outputs)
        loss.backward()
        
        optimizer.step()
        
        tr_running_loss += loss.item()
        
    
        predicted += outputs.detach().cpu().numpy().tolist()
        true += target.detach().cpu().numpy().tolist()
        
    tr_running_loss = tr_running_loss / len(dataloader)
    predicted = np.array(predicted)
    true = np.array(true)
        
    return model, tr_running_loss, true, predicted


# In[4]:


def train_step_by_avg(dataloader, model, criterion, optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    
    tr_running_loss = 0
    predicted = []
    true = []
 
    for data, target in dataloader:
    
        data = data.float()
        data = data.to(device)
        target= target.float()
        target = target.to(device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(data)
        outputs= outputs.view(-1,)
        
        loss = criterion(target, outputs)
        loss.backward()
        
        optimizer.step()
        
        tr_running_loss += loss.item()
        
    
        predicted += outputs.detach().cpu().numpy().tolist()
        true += target.detach().cpu().numpy().tolist()
        
    tr_running_loss = tr_running_loss / len(dataloader)
    predicted = np.array(predicted)
    true = np.array(true)
        
    return model, tr_running_loss, true, predicted


# In[5]:


def train_model(model, tr_loader, val_loader, criterion, model_path):

    num_geno= tr_loader.dataset.data.shape[1] - config.num_env
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Model is traning on: ', device)
    
    
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = np.inf
    best_pcc = -1
    no_improve = 0
    i = 0
    best_epoch = 0
    
    tr_losses = []
    val_losses = []
    while True:
        model, tr_loss, true, predicted = train_step(tr_loader, model, criterion, optimizer)
        tr_pcc = calculate_perf_measure(true, predicted)
        tr_avg_loc_pcc = calculate_loc_spec_pcc(tr_loader.dataset.trials, 
                                             tr_loader.dataset.locations,
                                             true,
                                             predicted
                                            )
        tr_losses.append(tr_loss)
        
        val_loss, val_true, val_predicted = evaluate_step(val_loader, model, criterion)
        val_pcc = calculate_perf_measure(val_true, val_predicted)
        val_avg_loc_pcc = calculate_loc_spec_pcc(val_loader.dataset.trials, 
                                             val_loader.dataset.locations,
                                             val_true,
                                             val_predicted
                                            )
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = i
            no_improve = 0
            
            torch.save(model.state_dict(), model_path)
            
        else:
            no_improve += 1
    
            
        print("Current epoch: {} Best epoch {}".format(i, best_epoch))
        print("Training Loss: {}, training PCC: {}, location_specific_pcc: {}".format(tr_loss, tr_pcc, tr_avg_loc_pcc))
        print("Validation Loss: {}, validation PCC: {}, location_specific_pcc: {}".format(val_loss, val_pcc, val_avg_loc_pcc))
        print()

        i +=1
        
        
        if no_improve == 10:
            break
            
    return tr_losses, val_losses


# In[6]:


def train_model_by_avg(model, tr_loader, val_loader, criterion, model_path):

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Model is traning on: ', device)
    
    
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    best_val_loss = np.inf
    best_pcc = -1
    no_improve = 0
    i = 0
    best_epoch = 0
    
    tr_losses = []
    val_losses = []
    tr_pccs =[]
    val_pccs =[]
    while True:
        model, tr_loss, true, predicted = train_step_by_avg(tr_loader, model, criterion, optimizer)
        tr_pcc = calculate_perf_measure(true, predicted)
        tr_losses.append(tr_loss)
        tr_pccs.append(tr_pcc)
        
        val_loss, val_true, val_predicted = evaluate_step_by_avg(val_loader, model, criterion)
        val_pcc = calculate_perf_measure(val_true, val_predicted)
        val_losses.append(val_loss)
        val_pccs.append(val_pcc)
        
        if i > 20:
            if best_pcc < val_pcc:
                best_pcc = val_pcc
                best_val_loss = val_loss
                best_epoch = i
                no_improve = 0

                torch.save(model.state_dict(), model_path)

            else:
                no_improve += 1

            if i % 200 == 0:
                clear_output(wait=True)
    
            
        print("Current epoch: {} Best epoch {}".format(i, best_epoch))
        print("Training Loss: {}, training PCC: {}".format(tr_loss, tr_pcc))
        print("Validation Loss: {}, validation PCC: {}".format(val_loss, val_pcc))
        # print(val_predicted)
        print()

        i +=1
        
        
        if no_improve == 100:
            break
            
    return tr_losses, val_losses, tr_pccs, val_pccs

