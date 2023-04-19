#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


def plot_scatter(x, y, xlabel, ylabel, xlim=None, ylim=None, axis_range='equal'):
    sns.scatterplot(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if axis_range != None:
        if xlim == None and axis_range == 'equal':
            xlim = np.amin(x.tolist() + y.tolist()) - 0.5 , np.amax(x.tolist() + y.tolist()) +0.5
            ylim =xlim
        else:
            xlim = np.amin(x) - 0.5 , np.amax(x) +0.5
            ylim = np.amin(y) - 0.5 , np.amax(y) +0.5

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])


# In[ ]:


def plot_histogram(x, xlabel, ylabel):
    sns.histplot(x, binwidth=0.1, binrange=(-1,1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

