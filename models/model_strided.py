#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:34:20 2019
Convolutional Autoencoder implemented as in Learning Design Semantics [UIST,2018]

with only conv, strided convolution for upsizing and downsizing the spatial dimension
@author: dipu
"""

import torch.nn as nn
import torch.nn.functional as F

class CAE_strided(nn.Module):
    
    def __init__ (self):
        super(CAE_strided, self).__init__()
        self.encoder = nn.Sequential(
                nn.Conv2d(3,8,3, stride=2),
                nn.ReLU(),
                nn.Conv2d(8,16,3, stride=2),
                nn.ReLU(),
                nn.Conv2d(16,16,3, stride=2),
                nn.ReLU(),
                nn.Conv2d(16,32,3, stride=2),
                nn.ReLU()
        )
                
        self.decoder = nn.Sequential(               
               nn.ConvTranspose2d(32,16,3, stride=2),
               nn.ReLU(),
               
               nn.ConvTranspose2d(16,16,3, stride=2),
               nn.ReLU(),
               
               nn.ConvTranspose2d(16,8,3, stride=2),
               nn.ReLU(),
               
               nn.ConvTranspose2d(8,3,3, stride=2),
               nn.ReLU(),
               )
        
#    def forward(self, x, training = True):
#        x_enc = self.encoder(x)
#        if not(training):
#            x_enc = x_enc.view(x_enc.size(0), -1)
#            return x_enc
#        else:
#            x_rec = self.decoder(x)
#            return x_enc, x_rec
                
    def forward(self, x):
        x = self.encoder(x)
        x_enc = x.view(x.size(0), -1)
        x_rec = self.decoder(x)
        
        return x_enc, x_rec
                           
            
