#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:34:38 2019
Deep Auto-Encoder for UI Retrieval
Dataset: RICO semantified UIs
@author: dipu
"""

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import pandas as pd 
import os
import numpy as np
from RICO_Dataset_plus_25ChannelOut import RICO_Dataset
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.model_CAE_emb512_25ChannelOut import ConvAutoEncoder

import errno
import os.path as osp
import shutil
import time 
import sys 

#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
        
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise        
#%% Data Preparation
data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations'
#data_dir =  '/home/dipu/dataset/

if not os.path.exists('/mnt/scratch/Dipu/RICO/UI_data.p'):
    # Read the filenames [*.png] and save into a list. Split Train and Validation Sets
    ui_names = [f for f in os.listdir(data_dir) if (os.path.isfile(os.path.join(data_dir, f)) &  (os.path.splitext(f)[1] == ".png"))]
    random_seed = 42
    
    dataset_size = len(ui_names)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    split = int(np.floor(0.8 * dataset_size))
    train_indices, test_indices =  indices[:split] , indices[split:]
    train_uis = [ui_names[x] for x in train_indices]
    test_uis = [ui_names[x] for x in test_indices]
    
    UI_data = {"ui_names": ui_names, "train_uis" : train_uis, "test_uis": test_uis}
    pickle.dump(UI_data, open("/mnt/scratch/Dipu/RICO/UI_data.p", "wb"))
else:
    UI_data = pickle.load(open('/mnt/scratch/Dipu/RICO/UI_data.p', 'rb'))
    train_uis = UI_data['train_uis'] 
#UI_data2 = pickle.load(open("/mnt/scratch/Dipu/RICO dataset/UI_data.p", "rb"))

nocomp_imlist = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/no_component_imglist.pkl', 'rb'))
ncomp_g100_imglist = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/ncomponents_g100_imglist.pkl', 'rb'))
nocomp_imlist = [x +'.png' for x in nocomp_imlist]
ncomp_g100_imglist = [x + '.png' for x in ncomp_g100_imglist]
train_uis = list(set(train_uis) - set(nocomp_imlist))
train_uis = list(set(train_uis) - set(ncomp_g100_imglist))

#%% Data Transforms and data loaders
BATCH_SIZE = 64
data_transform = transforms.Compose([
        transforms.Resize([254,126]),  #transforms.Resize([255,127])  # transforms.Resize([254,126])
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
   
train_dataset = RICO_Dataset(train_uis, data_dir, transform= data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, 
                                           drop_last = True, pin_memory=True, num_workers=16)

#%% Data Visualization
"""
def imshow(inp, title=None):
    # Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

imgs, names  = next(iter(train_loader))
out = torchvision.utils.make_grid(imgs)
imshow(out, title = [names])
"""

#%% Model and Training
model = ConvAutoEncoder()
model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

device = torch.device('cuda')
model.train()

epochs = 20
save_dir = '//home/dipu/codes/AutoEnconder_RicoDataset/runs/RICONew/modelCAE_emb512_25Channel_out' # 

torch.set_grad_enabled(True)
for epoch in range(epochs):
    losses = AverageMeter()
    s_ = time.time()
    
    for i , (data, names, img25Chan) in enumerate(train_loader):
        imgs = data.to(device)
        img25Chan = img25Chan.to(device)
        _, out = model(imgs)   #out = model(imgs) #enc, out = model(imgs)
        #img25Chan = F.interpolate(img25Chan, size= [239,111])
        
        loss = criterion(out, img25Chan)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.detach().item())
        
        if i%100 ==0:
           print( 'Epoch [%02d] [%05d / %05d] Average_Loss: %.3f' % (epoch+1, i*BATCH_SIZE, len(train_loader)*BATCH_SIZE, losses.avg ))
          #  print('Current Loss: ',loss)        
    
    if (epoch+1) % 5 == 0:
        state_dict = model.state_dict()
        
        # Save the model
        save_checkpoint({
            'state_dict': state_dict,
            'epoch': (epoch+1),
        }, is_best=False, fpath=osp.join(save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))
    
        scheduler.step()
    
    t = time.time() - s_
    print('1 training epoch takes %.2f hour' % (t/3600))

    
#TODO
""" 
1. Add Jitter/Noise
    A. Jitter transform data augmentation
    B. Add noise gaussian noise to the input
2. Train with ResNet based Encoder and Decoder 
3. Change the activation function at the output layer
4. Lessen max pooling layers, instead use convolutions with stride = 2 



    
"""        