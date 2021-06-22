#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:44:08 2020

@author: dipu
"""
from torchvision import transforms
import torch
import torch.nn as nn
import models
from RICO_Dataset import RICO_Dataset
from utils import AverageMeter, save_checkpoint, load_checkpoint, mkdir_if_missing

import argparse
import pickle
import time
import os
import os.path as osp

h = 48
w = 24
#class Autoencoder(nn.Module):
#    def __init__(self):
#        super(Autoencoder, self).__init__()
#        self.encoder = nn.Sequential(
#            nn.Linear(w * h * 3, 11200),
#            nn.ReLU(True),
#            nn.Linear(11200, 2048),
#            nn.ReLU(True), 
#            nn.Linear(2048, 256), 
#            nn.ReLU(True), 
#            nn.Linear(256, 64))
#        self.decoder = nn.Sequential(
#            nn.Linear(64, 256),
#            nn.ReLU(True),
#            nn.Linear(256, 2048),
#            nn.ReLU(True),
#            nn.Linear(2048, 11200), 
#            nn.ReLU(True), 
#            nn.Linear(11200, w * h * 3), 
#            nn.Tanh()
#            )
#    def forward(self, x):
#        x = x.view(x.size(0), -1)
#        x_enc = self.encoder(x)
#        x_rec = self.decoder(x_enc)
#        return x_enc, x_rec


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
#            nn.Linear(w * h * 3, 11200),
#            nn.ReLU(True),
            nn.Linear(w * h * 3, 2048),
            nn.ReLU(True), 
            nn.Linear(2048, 256), 
            nn.ReLU(True), 
            nn.Linear(256, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 2048),
            nn.ReLU(True),
#            nn.Linear(2048, 11200), 
#            nn.ReLU(True), 
            nn.Linear(2048, w * h * 3), 
            nn.Tanh()
            )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x_enc = self.encoder(x)
        x_rec = self.decoder(x_enc)
        return x_enc, x_rec




def main(args):
    print('OKAY.. Ready to Start.. Training ')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    #data_dir = '/mnt/amber/scratch/Dipu/RICO/MySemanticUI/'
    data_dir = '/mnt/amber/scratch/Dipu/RICO/MySemanticUIDefault/'
    
    
    UI_data = pickle.load(open('/mnt/amber/scratch/Dipu/RICO/UI_data.p', 'rb'))
    train_uis = UI_data['train_uis'] 
    
#    UI_test_data = pickle.load(open("/mnt/amber/scratch/Dipu/RICO/UI_test_data.p", "rb"))
#    query_uis = UI_test_data['query_uis']
#    gallery_uis = UI_test_data['gallery_uis']
    
    rico_info = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/rico_box_info.pkl', 'rb'))
    rico_ids = list(rico_info.keys())
    
    rico_ids = [x+'.png' for x in rico_ids]
    train_uis = list(set(rico_ids) & set(train_uis))
   
    
#    train_uis = [x.replace('.png', '') for x in train_uis]
#    query_uis = [x.replace('.png', '') for x in query_uis]
#    gallery_uis = [x.replace('.png', '') for x in gallery_uis]
 
#    rico_split_set2 = pickle.load(open(split_set_file, 'rb'))
#    train_uis = rico_split_set2['train_uis']
#    train_uis_fns = [x + '.png' for x in train_uis]

    BATCH_SIZE = args.batch_size
        
    data_transform = transforms.Compose([
            transforms.Resize([h,w]),  #transforms.Resize([255,127])  # transforms.Resize([254,126])
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
        
    # Dataset and Dataloader
    train_dataset = RICO_Dataset(train_uis, data_dir, transform= data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, 
                                           drop_last = True, pin_memory=True, num_workers=16)    
    


    
    # Model and Training
    device = torch.device('cuda')
    model = Autoencoder()
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 20 
    save_dir = 'runs_AE_MySemanticUI_h%s%s/%s/'%(h,w,args.model_name)
    
    model.train()
    torch.set_grad_enabled(True)
    for epoch in range(epochs):
        losses = AverageMeter()
        s_ = time.time()
        
        for i , (data, names) in enumerate(train_loader):
            data = data.view(data.size(0), -1)
            imgs = data.to(device)
            _, out = model(imgs)   #out = model(imgs) #enc, out = model(imgs)
            loss = criterion(out, imgs)
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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # optimization
    parser.add_argument('--batch_size', default = 256, type=int, metavar='N',  
                        help='mini-batch size (1 = pure stochastic) Default: 256') 
    # model
    parser.add_argument('--model_name', default = 'upsample_512', type = str,
                        help = 'which CNN autoencoder: upsample or strided or strided_512 or upsample_512')
    
    parser.add_argument('--gpu_id', type=str, default = '3', help = 'GPU ID')

    
    args = parser.parse_args()
    
    main(args)