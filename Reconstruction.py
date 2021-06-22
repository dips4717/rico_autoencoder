#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:06:56 2019
UI retrieval test
Plot retrived UIs
Plot the reconstructed UIs
@author: dipu
"""

import torch 
import numpy as np
#from model_CAE import ConvAutoEncoder
from model_CAE_2 import ConvAutoEncoder
from utils import load_checkpoint
import pickle 
from RICO_Dataset import RICO_Dataset
from torchvision import transforms
#from utils import imshow
import torchvision
from matplotlib import pyplot as plt
from utils import mkdir_if_missing

        
def imshow(inp1, inp2, filename, title=None):
    # Imshow for Tensor.
    inp1 = inp1.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp1 = std * inp1 + mean
    inp1 = np.clip(inp1, 0, 1)
    
    inp2 = inp2.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp2 = std * inp2 + mean
    inp2 = np.clip(inp2, 0, 1)
    
    if title is not None:
        plt.title(title)
        
    plt.subplot(2,1,1)
    plt.imshow(inp1)
    
    plt.subplot(2,1,2)
    plt.imshow(inp2)  
    plt.pause(0.001) # pause a bit so that plots are updated       
    plt.savefig(filename, dpi =300)
    
        
model_name = 'model_CAE2_OnlyConv_emb2912'
model_path = '/home/dipu/codes/stacked-autoencoder-pytorch/runs/{}/ckp_ep20.pth.tar'.format(model_name)
data_dir = '/mnt/scratch/Dipu/RICO dataset/semantic_annotations'
UI_data = pickle.load(open("/mnt/scratch/Dipu/RICO dataset/UI_data.p", "rb"))
data_transform = transforms.Compose([
        transforms.Resize([255,127]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ui_names, train_uis, test_uis  = UI_data['ui_names'], UI_data['train_uis'], UI_data['test_uis']    

BATCH_SIZE = 4
train_dataset = RICO_Dataset(train_uis, data_dir, transform= data_transform)
test_dataset  = RICO_Dataset(test_uis, data_dir, transform= data_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle = False, 
                                           drop_last = True, pin_memory=True, num_workers=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= BATCH_SIZE, shuffle = False, 
                                           drop_last = True, pin_memory=True, num_workers=16)


resume = load_checkpoint(model_path)
model = ConvAutoEncoder()
epoch = resume['epoch']
model.load_state_dict(resume['state_dict'])
model = model.cuda()
model.eval()

dataset = 'Train'
device = torch.device('cuda')
torch.set_grad_enabled(False)

if dataset == 'Train':
    loader = train_loader
else:
    loader = test_loader
    
for i , (data, names) in enumerate(loader):
        inputs = data.to(device)
        outputs = model(inputs)
        inputs  = inputs.cpu()
        outputs = outputs.cpu()
        inputs  = torchvision.utils.make_grid(inputs)
        outputs = torchvision.utils.make_grid(outputs)
        
        save_fig_dir = 'Results/Reconstruction-{}/{}/'.format(model_name, dataset)
        mkdir_if_missing(save_fig_dir)
        
        filename = '%s%d.png'%(save_fig_dir,i)
        
        imshow(inputs.cpu(), outputs.cpu(), filename) #,  title = ' '.join(names)
        
#        imshow(outputs.cpu(), title = [names])
        
        #torchvision.utils.save_image(inputs, 'Results/model_CAE_emb512' + in_fname ) 
        #torchvision.utils.save_image(outputs, 'saved_images/model_CAE_emb512' + re_fname )
               
        #input("Press Enter to continue...")

        if i == 20:
            break







