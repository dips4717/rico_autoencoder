#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:26:23 2019

# Compute the Pixel Accurracy between the query image and retrieved images.
# Two version of the evla metrics:
# 1.    Average Pix accuracy: for each  class(component/element) in query, 
        compute the pixAccs and average them
# 2.    Weighted Pix accuracy: for each class in query, compute the pixAccs. 
        Computed the weighted mean where weights are proportional to areas covered by the components

@author: dipu
"""
import numpy as np
from PIL import Image

def get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames):
    data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/' 
    allClasses = boundingBoxes.getClasses()
    classPixAcc = dict([(key, []) for key in allClasses])
    
    n_query = len(q_fnames)
    avgPixAccArray = np.zeros((n_query,5))
    weightedPixAccArray = np.zeros((n_query,5))
    
    for i in range((sort_inds.shape[0])):   #Iterate over all the query images 
        qImageName = q_fnames[i]
        q_img  =  Image.open(data_dir+qImageName+'.png').convert('RGB')
        qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
        
        for j in range(5):     # Iterate over top-5 retrieved images
    #        print ('\nQuery: ', i, 'Retrieved Image: ', j ) 
            
            rImageName = g_fnames[sort_inds[i][j]]
#            r_img =   Image.open(data_dir+rImageName+'.png').convert('RGB')
            rBBoxes = boundingBoxes.getBoundingBoxesByImageName(rImageName)
            
            tempPixAcc = []
            weights = []
            qClasses = list(set([d.classId for d in qBBoxes]))
            
            for c in qClasses:
    #            print(c)
    #            fig, ax = plt.subplots(2,2)
    #            plt.setp(ax,  xticklabels=[], yticklabels=[])
    #            fig.suptitle(c, fontsize=20)
    #            ax[0,0].imshow(q_img)        
    #            ax[1,0].imshow(r_img)
                
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
    #            ax[1,1].imshow(Image.fromarray(np.transpose(mask2)))   
    #            plt.show()
        
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
             
            avgPixAccArray[i][j] = np.mean(tempPixAcc) 
    
            weightTotal = np.sum(weights)
            weightvalues = np.divide(weights, weightTotal)
            weightedPixAccArray[i][j] = np.sum(tempPixAcc*weightvalues) 
            
        
    
    meanAvgPixAcc = np.mean(avgPixAccArray, axis=1)
    overallMeanAvgPixAcc = np.mean(meanAvgPixAcc)
    
    meanWeightedPixAcc = np.mean(weightedPixAccArray, axis=1)
    overallMeanWeightedPixAcc = np.mean(meanWeightedPixAcc)
    
    print('Completed computing Pixel Accuracies: {}/{}'.format(i+1,n_query)) 
    
    return overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc
    
    