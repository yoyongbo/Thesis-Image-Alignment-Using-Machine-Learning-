#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torchvision.models as models


# In[3]:


class FeatureExtraction(torch.nn.Module):
    def __init__(self, use_cuda =True, feature_extraction_cnn ='vgg', last_layer=''):
        super(FeatureExtraction, self).__init__()
        self.model = models.vgg16(pretrained=True)
        vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                                  'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
                                  'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1',
                                  'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
        if last_layer == '':
            last_layer = 'pool4'
        last_layer_idx = vgg_feature_layers.index(last_layer)
        
        self.model = nn.Sequential(*list(self.model.features.children())
                                  [:last_layer_idx+1])
        
        # freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        if use_cuda:
            self.model.cuda()
            
    def forward(self, img_batch):
        #img_batch = img_batch.cuda()
        return self.model(img_batch.cuda())
        
        
        


# In[4]:


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2),
                                   1)+epsilon,
                         0.5).unsqueeze(1).expand_as(feature)

        return torch.div(feature, norm)


# In[5]:


class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):

        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h*w)
        feature_B = feature_B.view(b, c, h*w).transpose(1, 2)

        # perform matrix multiplication
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h,
                                              w, h*w).transpose(2, 3).transpose(1, 2)

        return correlation_tensor.to(device="cuda")


# In[6]:


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(240, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# In[7]:


class CNNGeometric(nn.Module):
    def __init__(self, normalize_features=False, geometric_model='affine',                                    normalize_matches=False,
                 use_cuda=True, feature_extraction_cnn='vgg'):
#(self, geometric_model='affine', normalize_features=True, normalize_matches=True,
                 #use_cuda=True, feature_extraction_cnn='vgg'
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches

        self.FeatureExtraction = FeatureExtraction(use_cuda=self.use_cuda,
                                                   feature_extraction_cnn=feature_extraction_cnn)

        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        
        output_dim = 6

        self.FeatureRegression = FeatureRegression(output_dim,
                                                   use_cuda=self.use_cuda)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, tnf_batch):
        #print('normalize_features:',self.normalize_features)
        #print('cnn_geometric_model.py: do feature extraction')
        # do feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        #print('cnn_geometric_model.py: feature_A')
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        # normalize
        #print('cnn_geometric_model.py: normalize')
        if self.normalize_features:
            print('self.normalize_features')
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)
        # do feature correlation
        correlation = self.FeatureCorrelation(feature_A, feature_B)
        # normalize
        if self.normalize_features:
            correlation = self.FeatureL2Norm(self.ReLU(correlation))

        # do regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)
        #print('shape_model:',theta.shape)

        return theta


# In[ ]:




