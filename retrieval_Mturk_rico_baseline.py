#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:05:11 2020

@author: dipu
"""

import numpy as np
import json
import pickle
from scipy.spatial.distance import cdist
from perform_tests import getBoundingBoxes

from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from utils import mkdir_if_missing

boundingBoxes = getBoundingBoxes()

path = '/mnt/amber/scratch/Dipu/RICO/ui_layout_vectors/ui_layout_vectors'
vectors = np.load('%s/ui_vectors.npy'%(path))

with open( '%s/ui_names.json'%(path), 'rb') as f:
    ui_names_json = json.load(f)
ui_names = ui_names_json['ui_names']                                                              
ui_names = [x[:-4] for x in ui_names] 
model_name = 'model_CAE_emb2688' 

feat_file = 'features-{}.p'.format(model_name)
with open(feat_file, 'rb') as f:
    features = pickle.load(f)

g_fnames = features['g_fnames']
q_fnames = features['q_fnames']    
q_feat = features['q_feat']

qind = [ui_names.index(x) for x in q_fnames]

gnames = [x for x in g_fnames if (x in ui_names)]
print('Number of missing gallery images:', len(g_fnames) - len(gnames))
gind = [ui_names.index(x) for x in gnames]

q_feat = vectors[qind,:]
g_feat = vectors[gind,:]

distances = cdist(q_feat, g_feat, metric= 'euclidean')
sort_inds = np.argsort(distances)

q_fnames = q_fnames
g_fnames = gnames

overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
  

#%%
import shutil
base_img = 'AutoEncoder_Images/'
base_sui = 'AutoEncoder_Semantic_UIs/'

img_path = '/mnt/amber/scratch/Dipu/RICO/combined/'
sui_path = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'

for ii in  range(50):   #Iterate over all the query images 
    q_img = img_path + q_fnames[ii] + '.jpg'
    q_sui = sui_path + q_fnames[ii] + '.png'
    
    dest_dir_img = base_img + q_fnames[ii] + '/'
    dest_dir_sui = base_sui + q_fnames[ii] + '/'
    
    mkdir_if_missing(dest_dir_img)
    mkdir_if_missing(dest_dir_sui)
    
    shutil.copy(q_img, dest_dir_img + 'Query.jpg') 
    shutil.copy(q_sui, dest_dir_sui + 'Query.png')
    
    
    for jj in range(20):
        r_img = img_path + g_fnames[sort_inds[ii][jj]] + '.jpg'
        shutil.copy(r_img, dest_dir_img + '%02d'%(jj+1) + '.jpg'  )
        
        r_sui = sui_path + g_fnames[sort_inds[ii][jj]]  + '.png'
        shutil.copy(r_sui, dest_dir_sui + '%02d'%(jj+1) + '.png')






















    
    
    
    
    