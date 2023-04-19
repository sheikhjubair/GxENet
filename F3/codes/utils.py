#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../codes')
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import pandas as pd
import config
import pickle


# In[2]:


class wheat_dataset(Dataset):
    def __init__(self, file, weather_scaler=None, target_scaler=None):
        
        with open(file, 'rb') as pfile:
            self.all_data = pickle.load(pfile)
        
        self.all_data = self.all_data[self.all_data['Value'] < 10]
        self.all_data.sort_values(by=['Loc_no', 'trial'], inplace=True)
        
        self.locations = self.all_data['Loc_no'].to_numpy()
        self.trials = self.all_data['trial'].to_numpy()
        self.target = self.all_data['Value'].to_numpy()
        self.weather = np.array(self.all_data['weather'].tolist())
        self.weather_raw = np.array(self.all_data['weather'].tolist())
        self.target_raw = self.all_data['Value'].to_numpy()
        self.genotype = self.all_data.iloc[:, 3:-14]
        self.weather_scaler = weather_scaler
        self.target_scaler = target_scaler
        self.scale()
        self.data = np.concatenate((self.genotype, self.weather), axis = 1)
        
    def scale(self):
        
        if self.weather_scaler == None:
            self.weather_scaler = MinMaxScaler()
            self.weather = self.weather_scaler.fit_transform(self.weather)
            with open(config.weather_scaler, 'wb') as wfile:
                pickle.dump(self.weather_scaler, wfile)
        else:
            self.weather = self.weather_scaler.transform(self.weather)
            
        if self.target_scaler == None:
            self.target_scaler = MinMaxScaler()
            self.target = self.target_scaler.fit_transform(self.target.reshape(-1, 1)).reshape(-1,)
            with open(config.target_scaler, 'wb') as wfile:
                pickle.dump(self.target_scaler, wfile)
        else:
            self.target = self.target_scaler.transform(self.target.reshape(-1,1)).reshape(-1,)
            
        

    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, ind):
        return self.locations[ind], self.trials[ind], self.data[ind], self.target[ind], self.target_raw[ind]


# In[3]:


class wheat_dataset_by_df(Dataset):
    def __init__(self, df, weather_scaler=None, target_scaler=None):
        
        
        self.all_data = df
        
        self.all_data = self.all_data[self.all_data['Value'] < 20]
        self.all_data.sort_values(by=['Loc_no', 'trial'], inplace=True)
        
        self.locations = self.all_data['Loc_no'].to_numpy()
        self.trials = self.all_data['trial'].to_numpy()
        self.target = self.all_data['Value'].to_numpy()
        self.weather = np.array(self.all_data['weather'].tolist())
        self.weather_raw = np.array(self.all_data['weather'].tolist())
        self.target_raw = self.all_data['Value'].to_numpy()
        self.genotype = self.all_data.iloc[:, 3:-14]
        self.weather_scaler = weather_scaler
        self.target_scaler = target_scaler
        self.scale()
        self.data = np.concatenate((self.genotype, self.weather), axis = 1)
        
    def scale(self):
        
        if self.weather_scaler == None:
            self.weather_scaler = MinMaxScaler()
            self.weather = self.weather_scaler.fit_transform(self.weather)
        else:
            self.weather = self.weather_scaler.transform(self.weather)
            
        if self.target_scaler == None:
            self.target_scaler = MinMaxScaler()
            self.target = self.target_scaler.fit_transform(self.target.reshape(-1, 1)).reshape(-1,)
        else:
            self.target = self.target_scaler.transform(self.target.reshape(-1,1)).reshape(-1,)
            
        

    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, ind):
        return self.locations[ind], self.trials[ind], self.data[ind], self.target[ind], self.target_raw[ind]


# In[4]:


class wheat_dataset_by_geno_avg_one_hot(Dataset):
    def __init__(self, file, one_hot_enc=None, min_max_scaler = None):
        with open(file, 'rb') as pfile:
            all_data = pickle.load(pfile)

        all_data = all_data[all_data['Value'] < 10]


        all_data_groupby = all_data.groupby(by=['GID']).agg({
            'Value': lambda x: np.average(x),
        })

        all_data_groupby.reset_index(inplace=True)
        all_data = all_data.drop_duplicates(subset=['GID'])

        all_data = all_data.drop(['Value'],axis=1)
        all_data_joined = all_data.merge(all_data_groupby, on=['GID'])
        
        seq_data = pd.read_pickle('../genotypic_data/sequence_data.pkl')
        gid_df = all_data_joined[['GID', 'Value']]
        
        seq_data.reset_index(inplace=True)
        
        seq_data_training = gid_df.merge(seq_data, left_on=['GID'], right_on=['index'], how='left')
        x = seq_data_training.iloc[:, 3:].to_numpy()
        x = x.tolist()
        seq_data_training['seq'] = x
        
        self.enc = one_hot_enc
        if one_hot_enc == None:
            neucleotides = np.array(['A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M']).reshape(-1, 1)
            self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.enc.fit(np.array(neucleotides))
            
        z= seq_data_training.seq.apply(lambda x: self.enc.transform(np.array(x).reshape(-1,1)))
        z = np.array(z.tolist())
        self.data = z
        
        self.scaler = min_max_scaler
        
        if self.scaler == None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(seq_data_training['Value'].to_numpy().reshape(-1,1))
            
        
        self.target = self.scaler.transform(seq_data_training['Value'].to_numpy().reshape(-1,1)).reshape(-1,)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]


# In[ ]:


class wheat_dataset_by_geno_avg(Dataset):
    def __init__(self, file):
        with open(file, 'rb') as pfile:
            all_data = pickle.load(pfile)

        all_data = all_data[all_data['Value'] < 10]


        all_data_groupby = all_data.groupby(by=['GID']).agg({
            'Value': lambda x: np.average(x),
        })

        all_data_groupby.reset_index(inplace=True)
        all_data = all_data.drop_duplicates(subset=['GID'])

        all_data = all_data.drop(['Value'],axis=1)
        all_data_joined = all_data.merge(all_data_groupby, on=['GID'])
        
        self.data = all_data_joined.iloc[:, 3:-14].to_numpy()
        
        self.target = all_data_joined['Value'].to_numpy().reshape(-1,)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]


# In[ ]:


class wheat_dataset_by_env_avg(Dataset):
    def __init__(self, file):
        with open(file, 'rb') as pfile:
            all_data = pickle.load(pfile)

        all_data = all_data[all_data['Value'] < 10]


        all_data_groupby = all_data.groupby(by=['Loc_no', 'trial']).agg({
            'Value': lambda x: np.average(x),
        })

        all_data_groupby.reset_index(inplace=True)
        all_data = all_data.drop_duplicates(subset=['Loc_no', 'trial'])

        all_data = all_data.drop(['Value'],axis=1)
        all_data_joined = all_data.merge(all_data_groupby, on=['Loc_no', 'trial'])
        
        self.data = all_data_joined['weather'].tolist()
        self.data = np.array(self.data)
        self.target = all_data_joined['Value'].to_numpy().reshape(-1,)
        self.trial = all_data_joined['trial'].to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]


# In[ ]:


class wheat_dataset_by_representation(Dataset):
    def __init__(self, file, avg_by):
        with open(file, 'rb') as pfile:
            all_data = pickle.load(pfile)

        all_data = all_data[all_data['Value'] < 10]
        
        self.weather = all_data['weather'].tolist()
        self.weather = np.array(self.weather)
        
        self.genotypes = all_data.iloc[:, 3:-14].to_numpy()
        
        self.target = all_data['Value'].to_numpy()


        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]


# In[ ]:


class wheat_dataset_after_representation(Dataset):
    def __init__(self, representation_file, main_data_file):
        with open(representation_file, 'rb') as pfile:
            representation_data = pickle.load(pfile)
            
        with open(main_data_file, 'rb') as pfile:
            primary_data = pickle.load(pfile)
            primary_data = primary_data[primary_data['Value'] < 10]

        self.data = representation_data[:, :-1]
        self.target = representation_data[:,-1]
        
        #remove ---
#         with open('testing_scaler.pkl', 'rb') as wfile:
#             scaler = pickle.load(wfile)
        
#         self.target = scaler.transform(self.target.reshape(-1,1))
#         self.target = self.target.reshape(-1,)
        
        #---------
        
        self.locations = primary_data['Loc_no'].to_numpy()
        self.trials = primary_data['trial'].to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]


# In[9]:


class wheat_dataset_with_soil(Dataset):
    def __init__(self, representation_file, main_data_file, soil_file):
        with open(representation_file, 'rb') as pfile:
            representation_data = pickle.load(pfile)
            
        with open(main_data_file, 'rb') as pfile:
            primary_data = pickle.load(pfile)
            primary_data = primary_data[primary_data['Value'] < 10]
            
        with open(soil_file, 'rb') as pfile:
            soil_data = pickle.load(pfile)
        
        primary_data = primary_data.merge(soil_data, how='left', on = ['Loc_no', 'trial'])
        soil_info = np.array(primary_data['embedding'].tolist())

        self.data = representation_data[:, :-1]
        self.target = representation_data[:,-1]
        
        self.data = np.concatenate((self.data, soil_info), axis=1)
        
        self.locations = primary_data['Loc_no'].to_numpy()
        self.trials = primary_data['trial'].to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        return self.data[ind], self.target[ind]


# In[5]:


class loc_specific_wheat_dataset(Dataset):
    def __init__(self, loc_specific_data):
        
        
        self.all_data = loc_specific_data
        
        self.all_data = self.all_data[self.all_data['Value'] < 15]
        self.all_data.sort_values(by=['Loc_no', 'trial'], inplace=True)
        
        self.locations = self.all_data['Loc_no'].to_numpy()
        self.trials = self.all_data['trial'].to_numpy()
        self.target = self.all_data['normalized_value'].to_numpy()
        self.weather = np.array(self.all_data['normalized_weather'].tolist())
        self.weather_raw = np.array(self.all_data['weather'].tolist())
        self.target_raw = self.all_data['Value'].to_numpy()
        self.genotype = self.all_data.iloc[:, 3:-16]
    
        # self.scale()
        self.data = np.concatenate((self.genotype, self.weather), axis = 1)
        
#     def scale(self):
        
#         if self.weather_scaler == None:
#             self.weather_scaler = MinMaxScaler()
#             self.weather = self.weather_scaler.fit_transform(self.weather)
#         else:
#             self.weather = self.weather_scaler.transform(self.weather)
            
        # if self.target_scaler == None:
        #     self.target_scaler = MinMaxScaler()
        #     self.target = self.target_scaler.fit_transform(self.target.reshape(-1, 1)).reshape(-1,)
        # else:
        #     self.target = self.target_scaler.transform(self.target.reshape(-1,1)).reshape(-1,)
            
        

    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, ind):
        return self.locations[ind], self.trials[ind], self.data[ind], self.target[ind], self.target_raw[ind]


# In[6]:


class wheat_dataset(Dataset):
    def __init__(self, file, weather_scaler=None, target_scaler=None):
        
        with open(file, 'rb') as pfile:
            self.all_data = pickle.load(pfile)
        
        self.all_data = self.all_data[self.all_data['Value'] < 10]
        
        
        all_data_groupby = self.all_data.groupby(by=['GID']).agg({
            'Value': lambda x: np.average(x)
        })
        
        self.locations = self.all_data['Loc_no'].to_numpy()
        self.trials = self.all_data['trial'].to_numpy()
        self.target = self.all_data['Value'].to_numpy()
        self.weather = np.array(self.all_data['weather'].tolist())
        self.weather_raw = np.array(self.all_data['weather'].tolist())
        self.target_raw = self.all_data['Value'].to_numpy()
        self.genotype = self.all_data.iloc[:, 3:-14]
        self.weather_scaler = weather_scaler
        self.target_scaler = target_scaler
        self.scale()
        self.data = np.concatenate((self.genotype, self.weather), axis = 1)
        
    def scale(self):
        
        if self.weather_scaler == None:
            self.weather_scaler = MinMaxScaler()
            self.weather = self.weather_scaler.fit_transform(self.weather)
            with open(config.weather_scaler, 'wb') as wfile:
                pickle.dump(self.weather_scaler, wfile)
        else:
            self.weather = self.weather_scaler.transform(self.weather)
            
        if self.target_scaler == None:
            self.target_scaler = MinMaxScaler()
            self.target = self.target_scaler.fit_transform(self.target.reshape(-1, 1)).reshape(-1,)
            with open(config.target_scaler, 'wb') as wfile:
                pickle.dump(self.target_scaler, wfile)
        else:
            self.target = self.target_scaler.transform(self.target.reshape(-1,1)).reshape(-1,)
            
        

    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, ind):
        return self.locations[ind], self.trials[ind], self.data[ind], self.target[ind], self.target_raw[ind]


# In[7]:


def create_dataloader(file, weather_scaler, target_scaler, is_training = True):
    dataset = wheat_dataset(file, weather_scaler, target_scaler)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[8]:


def create_dataloader_by_data(data, is_training = True):
    dataset = loc_specific_wheat_dataset(data)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[9]:


def create_dataloader_by_df(data, weather_scaler, target_scaler, is_training = True):
    dataset = wheat_dataset_by_df(data)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[10]:


def create_dataloader_by_geno_avg(file, is_training = True, one_hot_enc=None, min_max_scaler = None):
    dataset = wheat_dataset_by_geno_avg(file)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[ ]:


def create_dataloader_by_env_avg(file, is_training = True, one_hot_enc=None, min_max_scaler = None):
    dataset = wheat_dataset_by_env_avg(file)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[ ]:


def create_dataloader_for_rep_learned_data(representation_file, main_data_file, is_training = True):
    dataset = wheat_dataset_after_representation(representation_file, main_data_file)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[ ]:


def create_dataloader_soil(representation_file, main_data_file, soil_file, is_training = True):
    dataset = wheat_dataset_with_soil(representation_file, main_data_file, soil_file)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[ ]:


def create_dataloader_by_geno_avg_one_hot(file, is_training = True, one_hot_enc=None, min_max_scaler = None):
    dataset = wheat_dataset_by_geno_avg_one_hot(file, one_hot_enc, min_max_scaler)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle= False)
    
    return dataloader


# In[11]:


def rescale_data(scaler, predicted):
    rescaled = scaler.inverse_transform(predicted.reshape(-1,1))
    
    return rescaled.reshape(-1,)


# In[12]:


def dump_pickle(obj, file):
    with open(file, 'wb') as wfile:
        pickle.dump(obj, wfile)


# In[13]:


def read_pickle(file):
    with open(file, 'rb') as rfile:
        obj = pickle.load(rfile)
        
    return obj


# In[2]:





# In[8]:




