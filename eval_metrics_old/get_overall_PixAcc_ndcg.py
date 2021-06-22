#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:46:42 2019

@author: dipu
"""

from PIL import Image
import numpy as np 

data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations/'

#%%
def dcg_at_k(r, k, method=1):  
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max
#%% Pixel Accuracy
# Compute the Pixel Accurracy between the query image and retrieved images.
# Two version of the evla metrics:
# 1. Average Pix accuracy: for each  class(component/element) in query, compute the pixAccs and average them
# 2. Weighted Pix accuracy: for each class in query, compute the pixAccs. Computed the weighted mean where weights are proportional to areas covered by the components

def get_overall_PixAcc_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames): 
    allClasses = boundingBoxes.getClasses()
    classPixAcc = dict([(key, []) for key in allClasses])
    
    aNdcg = np.empty((1,0),float)
    wNdcg =np.empty((1,0),float)
    
    aDCG = np.empty((1,0),float)
    wDCG =np.empty((1,0),float)
    
    for i in range((sort_inds.shape[0])):   #Iterate over all the query images 
        qImageName = q_fnames[i]
        q_img  =  Image.open(data_dir+qImageName+'.png').convert('RGB')
        qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
        
        accList = []
        weightedAccList = []
    
        for j in range(len(g_fnames)):     # Iterate over top-5 retrieved images
        #        print ('\nQuery: ', i, 'Retrieved Image: ', j ) 
            
            rImageName = g_fnames[sort_inds[i][j]]
#            r_img =   Image.open(data_dir+rImageName+'.png').convert('RGB')
            rBBoxes = boundingBoxes.getBoundingBoxesByImageName(rImageName)
            
            tempPixAcc = []
            weights = []
            qClasses = list(set([d.classId for d in qBBoxes]))
            
            for c in qClasses:          
                mask1 = np.zeros(q_img.size, dtype=np.uint8) 
                mask2 = np.zeros(q_img.size).astype(np.uint8) 
                
                c_qboxes = [b for b in qBBoxes if b.classId == c]
                c_rboxes = [b for b in rBBoxes if b.classId == c]
            
                for cqbox in c_qboxes:
                    bb = cqbox.getBoundingBox()
                    mask1[bb[0] : bb[0]+bb[2], bb[1] : bb[1]+bb[3]] = 1
                
                for crbox in c_rboxes:
                    bb = crbox.getBoundingBox()
                    mask2[bb[0]: bb[0]+ bb[2], bb[1] : bb[1]+bb[3]] = 1
        
                sum_n_ii = np.sum(np.logical_and(mask1, mask2))
                sum_t_i  = np.sum(mask1)    
                if (sum_t_i == 0):
                    pixAcc_c = 0
                else:
                    pixAcc_c = sum_n_ii / sum_t_i
                
                tempPixAcc.append(pixAcc_c)    
                weights.append(sum_t_i)
                
                # Accumuate the pixel acc for all classes    
                classPixAcc[c].append(pixAcc_c)    
        #            print('Class:', c, ': ', pixAcc_c)
             
            current_acc = np.mean(tempPixAcc) 
        
            weightTotal = np.sum(weights)
            weightvalues = np.divide(weights, weightTotal)
            current_weightedAcc = np.sum(tempPixAcc*weightvalues) 
            
            accList.append(current_acc)
            weightedAccList.append(current_weightedAcc)
        
        
        dcg_t =  dcg_at_k(accList,5)
        wdcg_t = dcg_at_k(weightedAccList,5)
        aDCG = np.append(aDCG,dcg_t)
        wDCG = np.append(wDCG,wdcg_t)
        
        aGain = ndcg_at_k(accList,5)
        wGain = ndcg_at_k(weightedAccList,5)
        
        aNdcg = np.append(aNdcg,aGain)
        wNdcg = np.append(wNdcg,wGain)
    
    avg_aNdcg = np.mean(aNdcg)
    avg_wNdcg = np.mean(wNdcg)    

    return aDCG, wDCG, avg_aNdcg, avg_wNdcg
