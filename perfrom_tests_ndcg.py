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

from eval_metrics.get_overall_IOU_ndcg import get_overall_IOU_ndcg
from eval_metrics.get_overall_ClasswiseIou_ndcg import get_overall_ClasswiseIou_ndcg
from eval_metrics.get_overall_PixAcc_ndcg import get_overall_PixAcc_ndcg

#%% Retrieval 

def main():
    boundingBoxes = getBoundingBoxes()
    models = ['model_CAE_emb2688', 'model_CAE_emb512', 'model_CAE2_OnlyConv_emb2912']
    for model_name in models:
        onlyGallery = True
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
    
        
        ndcgMeanIou, ndcgMeanWeightedIou   = get_overall_IOU_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames)         
        ndcgMeanClassIou, ndcgMeanWeightedClassIou = get_overall_ClasswiseIou_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames)
        ndcgMeanAvgPixAcc, ndcgMeanWeightedPixAcc = get_overall_PixAcc_ndcg(boundingBoxes,sort_inds,g_fnames,q_fnames)
    
    print('\n\nNDCG@5')
    print(model_name)
    print('GAlleryOnly Flag:', onlyGallery)
    print('The overallMeanIou = {:.3f}  '.format(ndcgMeanIou))
    print('The overallMeanWeightedIou = {:.3f}'.format(ndcgMeanWeightedIou))
    print('The overallMeanClassIou = {:.3f})'.format(ndcgMeanClassIou))
    print('The overallMeanWeightedClassIou = {:.3f})'.format(ndcgMeanWeightedClassIou))
    print('The overallMeanAvgPixAcc = {:.3f}'.format(ndcgMeanAvgPixAcc))
    print('The overallMeanWeightedPixAcc = {:.3f} '.format(ndcgMeanWeightedPixAcc))

#%%
def dcg_at_k(r, k, method=1):  
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


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
    
#    testBoundingBoxes(allBoundingBoxes)        
    return allBoundingBoxes


def testBoundingBoxes(boundingBoxes, samples = ['28970', '62918']):
    #Visualize if every colored element is plotted or not.
    from matplotlib import pyplot as plt
    from PIL import Image
    import matplotlib.patches as patches
    samples = ['28970', '62918']
    
    base_ui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
    base_im_path = '/mnt/scratch/Dipu/RICO/combined/'
    
    for sample in samples:
        img = base_ui_path + sample + '.png'
        img = Image.open(img).convert('RGB')
        img2 = base_im_path + sample + '.jpg'
        img2 = Image.open(img2).convert('RGB')
        
        fig, ax = plt.subplots(1,2)
        plt.setp(ax,  xticklabels=[], yticklabels=[])
        ax[0].imshow(img2)
        ax[1].imshow(img)
        bbs = boundingBoxes.getBoundingBoxesByImageName(sample)
        for bb in bbs:
            bb_cordinates = bb.getBoundingBox()
            bb_class = bb.classId
#            if bb_cordinates[2] < 0:
            rect = patches.Rectangle((bb_cordinates[0], bb_cordinates[1]), bb_cordinates[2], bb_cordinates[3], linewidth=2, edgecolor='r', facecolor= 'none')
            ax[1].add_patch(rect)
            ax[1].text(bb_cordinates[0], bb_cordinates[1],  bb_class,  fontsize=8, color= 'r', verticalalignment='top')
        plt.show()  

if __name__ == '__main__':
    main()





