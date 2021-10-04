#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils
from skimage import io, transform



# In[3]:


'''
torch.utils.data.Dataset is an abstract class representing a dataset. 
Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that 
dataset[i] can be used to get ith sample
'''
class Yong_DataSet(Dataset):
    def __init__(self,csv_file,training_image_path,geometric_model='affine',                                    transform=None):
        
        self.yong_data = pd.read_csv(csv_file)
        self.training_image_path = training_image_path
        self.transform = transform
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.csv_file = pd.read_csv('yong_data.csv')
        #self.root_dir = root_dir
        #delf.transform = transform
        
    def __len__(self):
        return len(self.yong_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        center_img_path = os.path.join(self.training_image_path, self.yong_data.iloc[idx,0])
        center_img = io.imread(center_img_path)
        trans_img_path = os.path.join(self.training_image_path, self.yong_data.iloc[idx,1])
        trans_img = io.imread(center_img_path)
        
        affine = self.yong_data.iloc[idx, 2:]
        affine= np.array([affine])
        affine = affine.astype('float').reshape(2,3)
        
        center_img = torch.Tensor(center_img.astype(np.float32))
        trans_img = torch.Tensor(trans_img.astype(np.float32))
        affine = torch.Tensor(affine.astype(np.float32))
        trans_img = trans_img.transpose(1, 2).transpose(0, 1)
        center_img = center_img.transpose(1, 2).transpose(0, 1)

        
        sample = {'source_image': center_img, 'target_image': trans_img, 'theta': affine}
        return sample
        


# In[ ]:




