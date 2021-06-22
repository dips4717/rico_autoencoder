#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:02:26 2019
Retrieval for 
@author: dipu
"""

import torch 
import numpy as np
import pickle 
from RICO_Dataset import RICO_Dataset
from torchvision import transforms
import torchvision
from matplotlib import pyplot as plt
from utils import mkdir_if_missing, load_checkpoint
import os 
from pyflann import FLANN
from scipy.spatial.distance import cdist
from PIL import Image
import time

from models.model_CAE import ConvAutoEncoder

def extract_features(data_loader, model):
    model.eval()
    torch.set_grad_enabled(False)
    features = []
    labels = []
   
    for i, (imgs, im_fn) in enumerate(data_loader):
        imgs = imgs.cuda()
        x_enc = model(imgs, training=False)
        outputs = x_enc.detach().cpu().numpy()
        features.append(outputs)
        labels += list(im_fn)
        print(i)
    return features, labels   
        
def plot_retrieved_images_and_uis(sort_inds, query_uis, gallery_uis, model_name):
    base_im_path = '/mnt/scratch/Dipu/RICO/combined/'
    base_ui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
    
    for i in  range((sort_inds.shape[0])): #range(1): 
        q_path = base_im_path + query_uis[i] + '.jpg'
        q_img  =  Image.open(q_path).convert('RGB')
        q_ui_path = base_ui_path + query_uis[i] + '.png'
        q_ui = Image.open(q_ui_path).convert('RGB')
        fig, ax = plt.subplots(2,6)
        plt.setp(ax,  xticklabels=[], yticklabels=[])
        fig.suptitle('Query-%s, %s (Gallery_Only-Set)'%(i, model_name), fontsize=20)
        #fig = plt.figure(1)
        fig.set_size_inches(30, 10) 
        #f1 = fig.add_subplot(2,6,1)
        
        ax[0,0].imshow(q_ui)
        ax[0,0].axis('off')
        ax[0,0].set_title('Query: %s '%(i) + query_uis[i] + '.png')
        ax[1,0].imshow(q_img)
        ax[1,0].axis('off') 
        ax[1,0].set_title('Query: %s '%(i) + query_uis[i] + '.jpg')
        #plt.pause(0.1)
     
        for j in range(5):
            path = base_im_path + gallery_uis[sort_inds[i][j]] + '.jpg'
           # print(gallery_uis[sort_inds[i][j]] )
            im = Image.open(path).convert('RGB')
            ui_path = base_ui_path + gallery_uis[sort_inds[i][j]] + '.png'
            #print(gallery_uis[sort_inds[i][j]]) 
            ui = Image.open(ui_path).convert('RGB')
            
            ax[0,j+1].imshow(ui)
            ax[0,j+1].axis('off')
            ax[0,j+1].set_title('Rank: %s '%(j+1) + gallery_uis[sort_inds[i][j]] + '.png')
            
            ax[1,j+1].imshow(im)
            ax[1,j+1].axis('off')
            ax[1,j+1].set_title('Rank: %s '%(j+1) + gallery_uis[sort_inds[i][j]] + '.jpg')
            
        directory =  'Retrieved_Images/{}/Gallery_Only/'.format(model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)  
            
        plt.savefig( directory + str(i) + '.png')
       # plt.pause(0.1)
        plt.close()
        #print('Wait')
        print(i)
        
   
#def main():
model_name = 'model_CAE_emb2688' #'model_CAE_emb512' #'model_CAE2_OnlyConv_emb2912'
model_path = '/home/dipu/codes/stacked-autoencoder-pytorch/runs/{}/ckp_ep20.pth.tar'.format(model_name)
data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations'

BATCH_SIZE = 128

UI_data = pickle.load(open("/mnt/scratch/Dipu/RICO/UI_data.p", "rb"))
ui_names  = UI_data['ui_names']
train_uis = UI_data['train_uis']
test_uis  = UI_data['test_uis'] 

if not(os.path.exists("/mnt/scratch/Dipu/RICO/UI_test_data.p")):
    random_seed = 42
    test_size = len(test_uis)
    indices = list(range(test_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    query_indices = indices[:50]
    gallery_indices = indices[50:]
    query_uis = [test_uis[x] for x in query_indices]
    gallery_uis = [test_uis[x] for x in gallery_indices]
    UI_test_data = {"query_uis": query_uis, "gallery_uis": gallery_uis}
    pickle.dump(UI_test_data, open("/mnt/scratch/Dipu/RICO/UI_test_data.p", "wb"))
else:
    UI_test_data = pickle.load(open("/mnt/scratch/Dipu/RICO/UI_test_data.p", "rb"))
    query_uis = UI_test_data['query_uis']
    gallery_uis = UI_test_data['gallery_uis']
    
data_transform = transforms.Compose([
        transforms.Resize([255,127]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])    
    

train_dataset = RICO_Dataset(train_uis, data_dir, transform= data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, 
                                           drop_last = True, pin_memory=True, num_workers=16)

query_dataset = RICO_Dataset(query_uis, data_dir, transform= data_transform)
query_loader = torch.utils.data.DataLoader(query_dataset, batch_size= BATCH_SIZE, shuffle=False,
                                           drop_last = False, pin_memory=True, num_workers=16)

gallery_dataset = RICO_Dataset(gallery_uis, data_dir, transform= data_transform)
gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size= BATCH_SIZE, shuffle=False,
                                           drop_last = True, pin_memory=True, num_workers=16)

model = ConvAutoEncoder()
resume = load_checkpoint(model_path)
model.load_state_dict(resume['state_dict'])
model = model.cuda()
model.eval()

feat_filename = 'features-' + model_name + '.p'
if not(os.path.exists(feat_filename)):
    q_feat, q_fnames = extract_features(query_loader, model)
    g_feat, g_fnames = extract_features(gallery_loader, model)
    t_feat, t_fnames = extract_features(train_loader, model)
    
    q_feat = np.concatenate(q_feat)
    g_feat = np.concatenate(g_feat)
    t_feat = np.concatenate(t_feat)
    
    features = {'q_feat': q_feat, 'q_fnames': q_fnames, 'g_feat': g_feat, 'g_fnames': g_fnames, 't_feat': t_feat,  't_fnames': t_fnames}
    
    pickle.dump(features, open(feat_filename, "wb"))
    print('Saved Features to %s\n'%(feat_filename))
else:
    feat_filename = 'features-' + model_name + '.p'
    features = pickle.load(open(feat_filename, "rb"))        
    q_feat = features["q_feat"]
    q_fnames = features["q_fnames"]
    g_feat = features["g_feat"]
    g_fnames = features["g_fnames"]
    t_feat = features["t_feat"]
    t_fnames = features["t_fnames"]


#r_feat = np.vstack((g_feat,t_feat))
#r_fnames = g_fnames + t_fnames

ts = time.time()
distances = cdist(q_feat, g_feat, metric= 'euclidean')
#distances = cdist(q_feat, r_feat, metric= 'euclidean')
sort_inds = np.argsort(distances) # each row in ascending order
time_elapsed = (time.time()-ts)/60
print('Time for search %s'%(time_elapsed))

plot_retrieved_images_and_uis(sort_inds, q_fnames, g_fnames, model_name)
#plot_retrieved_images_and_uis(sort_inds, q_fnames, r_fnames, model_name)


#flann = FLANN()
#results, dist = flann.nn(dataset,testset,2, algorithm="kmeans", branching="32", iterations=7, checks=16)



#if __name__ == "__main__":
#    main()
