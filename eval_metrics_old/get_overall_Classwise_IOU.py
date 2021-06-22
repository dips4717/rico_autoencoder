#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:44:15 2019

@author: dipu
"""
import numpy as np
from PIL import Image

#%% IoU Classwise

def get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames):
    data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
    n_query = len(q_fnames)
    
    avgClassIouArray = np.zeros((n_query,5))
    weightedClassIouArray = np.zeros((n_query,5))
    allClasses = boundingBoxes.getClasses()
    classwiseClassIou = dict([(key, []) for key in allClasses])
    
    for i in  range((sort_inds.shape[0])):   #Iterate over all the query images 
        qImageName = q_fnames[i]
        q_img  =  Image.open(data_dir+qImageName+'.png').convert('RGB')
        qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
        qClasses = list(set([d.classId for d in qBBoxes]))
        
        for j in range(5):     # Iterate over top-5 retrieved images
            rImageName = g_fnames[sort_inds[i][j]]
            rBBoxes = boundingBoxes.getBoundingBoxesByImageName(rImageName)
            
            iouTemp = []
            weights = []
            
            #Iterate over each element(boudingbox)
            for c in qClasses:                               # qbbs query bounding boxes
                mask1 = np.zeros(q_img.size, dtype=np.uint8) 
                mask2 = np.zeros(q_img.size).astype(np.uint8) 
                
                c_qboxes = [b for b in qBBoxes if b.classId == c]
                c_rboxes = [b for b in rBBoxes if b.classId == c]
            
                for cqbox in c_qboxes:
                    bb = cqbox.getBoundingBox()
                    mask1[bb[0] : bb[0]+bb[2], bb[1] : bb[1]+bb[3]] = 1
    #            ax[0,1].imshow(Image.fromarray(np.transpose(mask1)))
                
                for crbox in c_rboxes:
                    bb = crbox.getBoundingBox()
                    mask2[bb[0]: bb[0]+ bb[2], bb[1] : bb[1]+bb[3]] = 1
                
                intersec = np.sum(np.logical_and(mask1, mask2))
                union = np.sum(np.logical_or(mask1, mask2))
                iou_c = intersec/union
                
                iouTemp.append(iou_c)
                weights.append(np.sum(mask1))
    
                # Accumuate the pixel acc for all classes    
                classwiseClassIou[c].append(iou_c)  
            
            avgClassIouArray[i][j] = np.mean(iouTemp)
            
            weightTotal = np.sum(weights)
            weightvalues = np.divide(weights, weightTotal)
            weightedClassIouArray[i][j] = np.sum(iouTemp*weightvalues)
    
                    

    meanAvgPixAcc = np.mean(avgClassIouArray, axis=1)
    overallMeanClassIou = np.mean(meanAvgPixAcc)
    
    meanWeightedPixAcc = np.mean(weightedClassIouArray, axis=1)
    overallMeanWeightedClassIou = np.mean(meanWeightedPixAcc)
    
    print('Completed computing Classwise IoU: {}/{}'.format(i+1,n_query))
    
    return overallMeanClassIou, overallMeanWeightedClassIou, avgClassIouArray    
    
   