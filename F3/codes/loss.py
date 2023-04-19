#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from torch import nn


# In[ ]:


class mse_pcc_loss(nn.Module):
    def __init__(self):
        super(mse_pcc_loss, self).__init__()
        
    def forward(self, true, predicted):
        pcc = 0
#         x = predicted
#         y = true

#         vx = x - torch.mean(x)
#         vy = y - torch.mean(y)

#         pcc = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 0.0000001)
       
        
        mse = F.mse_loss(predicted, true)
        # if torch.isnan(pcc):
        #     pcc = -1
        total_cost = mse
        
        if torch.isnan(total_cost):
            print('true: ', true)
            print('pred: ', predicted)
            
        return total_cost


# In[ ]:


class pcc_loss(nn.Module):
    def __init__(self):
        super(pcc_loss, self).__init__()
        
    def forward(self, true, predicted):
        x = predicted
        y = true

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 0.0000001)
        cost = 1 - cost
        
        if torch.isnan(cost):
            print('true: ', true)
            print('pred: ', predicted)
        return cost
                                     


# In[ ]:


def calculate_perf_measure(true, predicted):
    pcc, pval = pearsonr(true, predicted)
    # print(true)
    # print(predicted)
    return pcc


# In[ ]:


def calculate_loc_spec_pcc(trials, locations, true, predicted):
    unique_trials = np.unique(trials)
    unique_locations = np.unique(locations)
    
    df = pd.DataFrame({
        'trials': trials,
        'locations': locations,
        'true': true,
        'predicted': predicted
    })
    
    total_pcc = 0
    count = 0
    for trial in unique_trials:
        for location in unique_locations:
            partial_df = df[(df['trials'] == trial) & (df['locations'] == location)]
            
            if partial_df.shape[0] > 5:
                pcc = calculate_perf_measure(partial_df['true'].to_numpy(),
                                      partial_df['predicted'].to_numpy())
                
                total_pcc += pcc
                count += 1
    
    avg_pcc = total_pcc / count
    
    return avg_pcc
            
            
        


# In[ ]:


def get_res_by_location(test_data):
    locations = test_data['Loc_no'].unique()
    trials = test_data['trial'].unique()

    result_dict ={}
    result_dict['location'] = []
    result_dict['trial'] = []
    result_dict['num_geno'] = []
    result_dict['pcc'] = []

    for location in locations:
        for trial in trials:
            partial_test = test_data[(test_data['Loc_no'] == location) & (test_data['trial'] == trial)]

            if len(partial_test) > 20:
               
                pcc = pearsonr(partial_test['Value'].to_numpy().reshape(-1,), partial_test['predicted'].to_numpy().reshape(-1,))[0]
                result_dict['location'].append(location)
                result_dict['trial'].append(trial)
                result_dict['num_geno'].append(len(partial_test))
                result_dict['pcc'].append(pcc)
   
    result_df = pd.DataFrame(result_dict)
   
    return result_df


# In[ ]:


class mse_pcc_weight_loss(nn.Module):
    def __init__(self):
        super(mse_pcc_loss, self).__init__()
        
    def forward(self, true, predicted, loc_no):
        pcc = 0
        unique_loc = torch.unique(loc_no)
        
        total_cost = 0
        for loc in unique_loc:
            ind = loc_no == loc
            partial_target = true[ind]
            partial_predicted = predicted[ind]
#         x = predicted
#         y = true

            vx = partial_target - torch.mean(partial_target)
            vy = partial_predicted - torch.mean(partial_predicted)

            pcc = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 0.0000001)
            pcc_imp = 1 - pcc
            
            partial_mse = F.mse_loss(partial_target, partial_predicted)
            total_cost = total_cost + partial_mse * pcc_imp
       
            
        return total_cost


# In[ ]:


def get_res_by_trial(test_data):
    locations = test_data['Loc_no'].unique()
    trials = test_data['trial'].unique()

    result_dict ={}
    
    result_dict['trial'] = []
    result_dict['num_geno'] = []
    result_dict['pcc'] = []

    
    for trial in trials:
        partial_test = test_data[test_data['trial'] == trial]

        if len(partial_test) > 20:
            
            pcc = pearsonr(partial_test['Value'].to_numpy().reshape(-1,), partial_test['predicted'].to_numpy().reshape(-1,))[0]
            
            result_dict['trial'].append(trial)
            result_dict['num_geno'].append(len(partial_test))
            result_dict['pcc'].append(pcc)
   
    result_df = pd.DataFrame(result_dict)
   
    return result_df

