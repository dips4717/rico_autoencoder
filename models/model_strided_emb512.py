#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:34:20 2019
with only conv, strided convolution for upsizing and downsizing the spatial dimension
@author: dipu
"""

import torch.nn as nn
import torch.nn.functional as F

class CAE_strided_dim512(nn.Module):
    
    def __init__ (self):
        super(CAE_strided_dim512, self).__init__()
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
        
        self.FC1 = nn.Linear(32*15*7, 512)
        self.FC2 = nn.Linear(512, 32*15*7)        
        
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
        x_enc = self.encoder(x)
        x_enc = x_enc.view(x_enc.size(0), -1)
        x_enc = self.FC1(x_enc)
        
        x_rec = self.FC2(x_enc)
        x_rec = x_rec.reshape(x_rec.size(0),32,15,7)            
        x_rec = self.decoder(x_rec)
        return x_enc, x_rec
                           
            
