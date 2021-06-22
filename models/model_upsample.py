#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:34:20 2019
Convolutional Autoencoder implemented as in Learning Design Semantics [UIST,2018]
@author: dipu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE_upsample(nn.Module):
    
    def __init__ (self):
        super(CAE_upsample, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3,8,3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(8,16,3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(16,16,3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(16,32,3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
        )
                
        self.decoder = nn.Sequential(
               nn.Upsample(scale_factor =2, mode = 'nearest'),
               nn.ConvTranspose2d(32,16,3),
               nn.ReLU(),
               
               nn.Upsample(scale_factor =2, mode = 'nearest'),
               nn.ConvTranspose2d(16,16,3),
               nn.ReLU(),
               
               nn.Upsample(scale_factor =2, mode = 'nearest'),
               nn.ConvTranspose2d(16,8,3),
               nn.ReLU(),
               
               nn.Upsample(scale_factor =2, mode = 'nearest'),
               nn.ConvTranspose2d(8,3,3),
               nn.ReLU(),
               )
        
#    def forward(self, x, training=True):
#        x_enc = self.encoder(x)
#        if not(training):
#            x_enc = x_enc.view(x_enc.size(0), -1)
#            return x_enc
#        else:
#            x_rec = self.decoder(x_enc)
#            return x_enc, x_rec
        
    def forward(self, x):
        x = self.encoder(x)
        x_enc = x.view(x.size(0), -1)
            
        x_rec = self.decoder(x)
        return x_enc, x_rec
                
                
            
        
#        self.conv1 = nn.Conv2d(3,8,3)
#        self.conv2 = nn.Conv2d(8,16,3)
#        self.conv3 = nn.Conv2d(16,16,3)
#        self.conv4 = nn.Conv2d(16,32,3)
#        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode = True)
#        self.relu = nn.ReLU(True)
#        
#        self.t_conv1 = nn.ConvTranspose2d(8,3,3)
#        self.t_conv2 = nn.ConvTranspose2d(16,8,3)
#        self.t_conv3 = nn.ConvTranspose2d(16,16,3)
#        self.t_conv4 = nn.ConvTranspose2d(32,16,3)
#        self.upsample = nn.Upsample(scale_factor =2, mode = 'nearest')
#        
#
#
#    def forward(self,x):
#        x1 = self.conv1(x)
#        x1 = self.relu(x1)
#        x1 = self.maxpool(x1)
#        print ('x1.shape', x1.shape)
#        
#        x2 = self.conv2(x1)
#        x2 = self.relu(x2)
#        x2 = self.maxpool(x2)
#        
#        x3 = self.conv3(x2)
#        x3 = self.relu(x3)
#        x3 = self.maxpool(x3)        
#        
#        x4 = self.conv4(x3)
#        x4 = self.relu(x4)
#        x4 = self.maxpool(x4)
#        print('x4.shape', x4.shape)
#
#        
#        if self.training:
#            x4_ = self.upsample(x4)
#            x4_ = self.t_conv4(x4_)
#            x4_ = self.relu(x4_)
#            
#            x3_ = self.upsample(x4_)
#            x3_ = self.t_conv3(x3_)
#            x3_ = self.relu(x3_)
#            
#            x2_ = self.upsample(x3_)
#            x2_ = self.t_conv2(x2_)
#            x2_ = self.relu(x2_)
#            
#            x1_ = self.upsample(x2_)
#            x1_ = self.t_conv1(x1_)
#            x1_ = self.relu(x1_)
#            
#        print ('Shape of x1_', x1_.shape)
#
#        return x1_
