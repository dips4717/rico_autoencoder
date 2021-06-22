#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:56:54 2019

@author: dipu
"""

import os 
import sys 
import glob
import json
from collections import defaultdict
from utils import extract_features, compute_iou
import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from PIL import Image
from matplotlib import pyplot as plt

import pickle
import time
from pyflann import FLANN
from scipy.spatial.distance import cdist
import numpy as np

global counter 
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
currentPath = os.path.dirname(os.path.realpath(__file__))
# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, '..', '..', 'lib')
add_path(libPath)

#%%
def dcg_at_k(r, k, method=1):  
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

#%% Ploting retrievals
def plot_retrieved_images_and_uis(sort_inds,q_fnames, g_fnames, model_name, \
                                  avgIouArray, weightedIouArray, \
                                  avgClassIouArray, weightedClassIouArray, \
                                  avgPixAccArray, weightedPixAccArray):
    from PIL import Image
    
    base_im_path = '/mnt/scratch/Dipu/RICO/combined/'
    base_ui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
    
    for i in range((sort_inds.shape[0])): #range(1): 
#    for i in  range(2):    
        q_path = base_im_path + q_fnames[i] + '.jpg'
        q_img  =  Image.open(q_path).convert('RGB')
        q_ui_path = base_ui_path + q_fnames[i] + '.png'
        q_ui = Image.open(q_ui_path).convert('RGB')
        
        fig, ax = plt.subplots(2,6, figsize=(30, 12), constrained_layout=True)
        plt.setp(ax,  xticklabels=[],  yticklabels=[])
        fig.suptitle('Query-%s, %s (Gallery_Only-Set)'%(i, model_name), fontsize=20)
        fig = plt.figure(1)
#        fig.set_size_inches(30, 12)
#        plt.subplots_adjust(bottom = 0.1, top=10)
        #f1 = fig.add_subplot(2,6,1)
        
        ax[0,0].imshow(q_ui)
        ax[0,0].axis('off')
        ax[0,0].set_title('Query: %s '%(i) + q_fnames[i] + '.png')
        ax[1,0].imshow(q_img)
        ax[1,0].axis('off') 
        ax[1,0].set_title('Query: %s '%(i) + q_fnames[i] + '.jpg')
        #plt.pause(0.1)
     
        for j in range(5):
            path = base_im_path + g_fnames[sort_inds[i][j]] + '.jpg'
           # print(g_fnames[sort_inds[i][j]] )
            im = Image.open(path).convert('RGB')
            ui_path = base_ui_path + g_fnames[sort_inds[i][j]] + '.png'
            #print(g_fnames[sort_inds[i][j]]) 
            ui = Image.open(ui_path).convert('RGB')
            
            ax[0,j+1].imshow(ui)
            ax[0,j+1].axis('off')
            ax[0,j+1].set_title('Rank: %s  '%(j+1)  + g_fnames[sort_inds[i][j]] \
              + '.png\nAvg IoU: %.3f'%(avgIouArray[i][j])+  '\nWeighted IoU: %.3f'%(weightedIouArray[i][j])\
              + '\nAvg Classwise IoU: %.3f'%(avgClassIouArray[i][j])+  '\nWeighted Classwise IoU: %.3f'%(weightedClassIouArray[i][j]) \
              + '\nAvg PixAcc: %.3f'%(avgPixAccArray[i][j])+  '\nWeighted PixAcc: %.3f'%(weightedPixAccArray[i][j])) 
    
            
            ax[1,j+1].imshow(im)
            ax[1,j+1].axis('off')
            ax[1,j+1].set_title('Rank: %s  '%(j+1) + g_fnames[sort_inds[i][j]] + '.jpg')
            
#        directory =  'Retrieval_Results_Iou_PixAcc/{}/Gallery_Only/'.format(model_name)
        directory =  'Retrieval_Results_Iou_Class_iou_PixAcc/{}/Gallery_Only/'.format(model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)  
            
        plt.savefig( directory + str(i) + '.png')
       # plt.pause(0.1)
        plt.close()
        #print('Wait')
        print('Plotting the retrieved images: {}'.format(i))

#plot_retrieved_images_and_uis(sort_inds,q_fnames, g_fnames, model_name, avgIouArray, weightedIouArray, avgClassIouArray, weightedClassIouArray, avgPixAccArray, weightedPixAccArray)

#%% Preparing the dataset
def parse_ui_elements(sui):
    """
    Parse the json file iteratively using recursion,, un winding all the nested chilfre
    returns the dictionay of elements   
    """
    
    global counter
    counter = 0
    elements = defaultdict(dict)
    
    def recurse(sui):
        global counter
        n_uis = len(sui['children'])
        for i in range(n_uis):                
            [x1, y1, x2, y2] = sui['children'][i]['bounds']
            elements[counter]['component_Label'] = sui['children'][i]['componentLabel']
            elements[counter]['x'] = x1
            elements[counter]['y'] = y1
            elements[counter]['w'] = x2-x1
            elements[counter]['h'] = y2-y1
            elements[counter]['iconClass'] = sui['children'][i].get('iconClass') 
            elements[counter]['textButtonClass'] = sui['children'][i].get('textButtonClass')
            
            counter +=1
            
            if sui['children'][i].get('children') != None:
                recurse(sui['children'][i])
    
    recurse(sui)        
    return elements, counter 

       
def getBoundingBoxes():
    allBoundingBoxes = BoundingBoxes()
    
    files = glob.glob(data_dir+ "*.json")
    for file in files:
        imageName = os.path.split(file)[1]
        imageName = imageName.replace(".json", "")
        print(imageName)
        
        with open(file, "r") as f:
           sui = json.load(f)   # sui = semantic ui annotation.
           
        elements, count = parse_ui_elements(sui)
        for i in range(count):
            box = elements[i]
            bb = BoundingBox(
                imageName,
                box['component_Label'],
                box['x'],
                box['y'],
                box['w'],
                box['h'],
                iconClass=box['iconClass'],
                textButtonClass=box['textButtonClass'])
            allBoundingBoxes.addBoundingBox(bb)  
            
    return allBoundingBoxes

#boundingBoxes = getBoundingBoxes()

#%% Retrieval 
onlyGallery = True
model_name = 'model_CAE2_OnlyConv_emb2912' # 'model_CAE_emb512' # 'model_CAE_emb2688' #   
feat_filename = 'features-' + model_name + '.p'
feat_filename = 'features-' + model_name + '.p'
features = pickle.load(open(feat_filename, "rb"))        
q_feat = features["q_feat"]
q_fnames = features["q_fnames"]
g_feat = features["g_feat"]
g_fnames = features["g_fnames"]

if not(onlyGallery):
    t_feat = features["t_feat"]
    t_fnames = features["t_fnames"]
    g_feat = np.vstack((g_feat,t_feat))
    g_fnames = g_fnames + t_fnames 
    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)
else:
    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)
    

#%% IoU 
# Compute the Intersection over Union between the query and retrieved images 
# For each query, iterate over all the bounding boxes one at a time. On each query, get all the elements that belong to the same class as the bbox in the query.
# For all the elements, compute the IoU, and select the best matched element with max Iou, detected element in the query image.  
      
avgIouArray = np.zeros((50,5))
weightedIouArray = np.zeros((50,5))
allClasses = boundingBoxes.getClasses()
classIou = dict([(key, []) for key in allClasses])

for i in  range((sort_inds.shape[0])):   #Iterate over all the query images 
    
    qImageName = q_fnames[i]
    qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
    
    for j in range(5):     # Iterate over top-5 retrieved images
        rImageName = g_fnames[sort_inds[i][j]]
        rBBoxes = boundingBoxes.getBoundingBoxesByImageName(rImageName)
        
        iouTemp = []
        weights = []
        
        #Iterate over each element(boudingbox)
        for bb in qBBoxes:                              # qbbs query bounding boxes
            bb_cordinates = bb.getBoundingBox()              
            bb_class = bb.classId 
            
            #get the bouding box in retrieved image that has same class
            rbbs = [d for d in rBBoxes if d.classId == bb_class]
            
            iouMax = 0 #sys.float_info.min
            for rbb in rbbs:
                assert(rbb.classId == bb_class)
                rbb_cordinates = rbb.getBoundingBox()
                iou =  compute_iou(bb_cordinates, rbb_cordinates)
                if iou > iouMax:
                    iouMax = iou
            assert(iou>=0)        
            #Store iou with best matched component label
            iouTemp.append(iouMax)
            weights.append(bb_cordinates[2]*bb_cordinates[3])
            # Update iou into coressponding ClassIou
            classIou[bb_class].append(iouMax)
            
        avgIouArray[i][j] = np.mean(iouTemp)     # Average Iou between a query and a retrieved image
        
        weightTotal = np.sum(weights)
        weights = np.divide(weights, weightTotal)
        weightedIou = sum(iouTemp*weights) 
        weightedIouArray[i][j] = weightedIou
    print('Computing IoU metric: {}/{}'.format(i,50))


#               iou =  compute_iou(bb_cordinates, rbb_cordinates)
#                iouTemp.append(iou)
                
#                # plot images
#                from matplotlib import pyplot as plt
#                from PIL import Image
#                import matplotlib.patches as patches
#                
#                
#                base_ui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
#                q_img = base_ui_path + qImageName + '.png'
#                r_img = base_ui_path + rImageName + '.png'
#                
#                fig, ax = plt.subplots(1,2)
#                plt.setp(ax,  xticklabels=[], yticklabels=[])
#                q_ui = Image.open(q_img).convert('RGB')
#                r_ui = Image.open(r_img).convert('RGB')
#                
#                q_rect = patches.Rectangle((bb_cordinates[0], bb_cordinates[1]), bb_cordinates[2], bb_cordinates[3], linewidth=2, edgecolor='r', facecolor= 'none')
#                r_rect = patches.Rectangle((rbb_cordinates[0], rbb_cordinates[1]), rbb_cordinates[2], rbb_cordinates[3], linewidth=2, edgecolor='r', facecolor= 'none')
#                
#                
#                ax[0].imshow(q_ui)
#                ax[0].axis('off') 
#                ax[0].add_patch(q_rect)
#                ax[0].text(bb_cordinates[0], bb_cordinates[1],  bb_class,  fontsize=8, color= 'r', verticalalignment='top')
#                
#                ax[1].imshow(r_ui)
#                ax[1].axis('off') 
#                ax[1].add_patch(r_rect)
#                ax[1].text(rbb_cordinates[0], rbb_cordinates[1],  rbb.classId,  fontsize=8, color= 'r', verticalalignment='top')
#                print(1)
            
#%% Pixel Accuracy
# Compute the Pixel Accurracy between the query image and retrieved images.
# Two version of the evla metrics:
# 1. Average Pix accuracy: for each  class(component/element) in query, compute the pixAccs and average them
# 2. Weighted Pix accuracy: for each class in query, compute the pixAccs. Computed the weighted mean where weights are proportional to areas covered by the components

allClasses = boundingBoxes.getClasses()
classPixAcc = dict([(key, []) for key in allClasses])
avgPixAccArray = np.zeros((50,5))
weightedPixAccArray = np.zeros((50,5))

for i in range((sort_inds.shape[0])):   #Iterate over all the query images 
    qImageName = q_fnames[i]
    q_img  =  Image.open(data_dir+qImageName+'.png').convert('RGB')
    qBBoxes = boundingBoxes.getBoundingBoxesByImageName(qImageName) 
    
    for j in range(5):     # Iterate over top-5 retrieved images
#        print ('\nQuery: ', i, 'Retrieved Image: ', j ) 
        
        rImageName = g_fnames[sort_inds[i][j]]
        r_img =   Image.open(data_dir+rImageName+'.png').convert('RGB')
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
        
    print('Computing Pixel Accuracies: {}/{}'.format(i,50))    

#%% IoU Classwise
avgClassIouArray = np.zeros((50,5))
weightedClassIouArray = np.zeros((50,5))
allClasses = boundingBoxes.getClasses()
classwiseClassIou = dict([(key, []) for key in allClasses])

for i in  range((sort_inds.shape[0])):   #Iterate over all the query images 
    qImageName = q_fnames[i]
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

    print('Computing Classwise IoU: {}/{}'.format(i,50))            
            
    
#%% Plotting the retrieved images with Iou and PixAcc metrics
#plot_retrieved_images_and_uis(sort_inds, avgIouArray, weightedIouArray, q_fnames, g_fnames, model_name)
#plot_retrieved_images_and_uis(sort_inds,q_fnames, g_fnames, model_name, avgIouArray, weightedIouArray, avgClassIouArray, weightedClassIouArray, avgPixAccArray, weightedPixAccArray)            

def compute_meanIou(avgIouArray, weightedIouArray):
    meanAvgIou = np.mean(avgIouArray, axis=1)
    overallMeanIou = np.mean(meanAvgIou)
    
    meanWeightedIou = np.mean(weightedIouArray, axis=1)    
    overallMeanWeightedIou = np.mean(meanWeightedIou)
    
    return overallMeanIou, overallMeanWeightedIou
    
def compute_meanPixelAcc(avgPixAccArray, weightedPixAccArray) :
    meanAvgPixAcc = np.mean(avgPixAccArray, axis=1)
    overallMeanAvgPixAcc = np.mean(meanAvgPixAcc)
    
    meanWeightedPixAcc = np.mean(weightedPixAccArray, axis=1)
    overallMeanWeightedPixAcc = np.mean(meanWeightedPixAcc)
    
    return overallMeanAvgPixAcc, overallMeanWeightedPixAcc

overallMeanIou, overallMeanWeightedIou = compute_meanIou(avgIouArray, weightedIouArray)
overallMeanClassIou, overallMeanWeightedClassIou = compute_meanIou(avgClassIouArray, weightedClassIouArray)
overallMeanAvgPixAcc, overallMeanWeightedPixAcc = compute_meanPixelAcc(avgPixAccArray, weightedPixAccArray)

print('\n', model_name)
print('GAlleryOnly Flag:', onlyGallery)
print('The overallMeanIou = {:.3f}  '.format(overallMeanIou))
print('The overallMeanWeightedIou = {:.3f}'.format(overallMeanWeightedIou))
print('The overallMeanClassIoU = {:.3f})'.format(overallMeanClassIou))
print('The overallMeanWeightedClassIoU = {:.3f})'.format(overallMeanWeightedClassIou))
print('The overallMeanAvgPixAcc = {:.3f}'.format(overallMeanAvgPixAcc))
print('The overallMeanWeightedPixAcc = {:.3f} '.format(overallMeanWeightedPixAcc))

#classwiseClassIou
#classPixAcc
#classIou

#%%
#TODO
#1. Normalized Discounted Cummulative Gain. for all models.
#2. Classwise pixel accuracies
#3. classwises average iou










