#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tqdm import tqdm


# In[4]:


def train(epoch, model, loss_fn, optimizer, dataloder,pair_generation_tnf):
    """
    Main function for training

    :param epoch: int, epoch index
    :param model: pytorch model object
    :param loss_fn: loss function of the model
    :param optimizer: optimizer of the model
    :param dataloader: DataLoader object
    """
    model.train()
    train_loss =0
    for batch_idx, batch in enumerate(tqdm(dataloder, desc='Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        #batch = pair_generation_tnf(batch)
        
        theta = model(batch)
        print()
        loss= loss_fn(theta, batch['theta'])
        
        
        
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy().item()
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    return train_loss
        