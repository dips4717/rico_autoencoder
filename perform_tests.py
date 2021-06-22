#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:01:07 2019

@author: dipu
"""
import os 

import glob
import json
from collections import defaultdict
from utils import extract_features, compute_iou
import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from PIL import Image
import pickle
from scipy.spatial.distance import cdist
import numpy as np

from eval_metrics.get_overall_IOU import get_overall_IOU
from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc

data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'

#def add_path(path):
#    if path not in sys.path:
#        sys.path.insert(0, path)
#

#currentPath = os.path.dirname(os.path.realpath(__file__))
## Add lib to PYTHONPATH
#libPath = os.path.join(currentPath, '..', '..', 'lib')
#add_path(libPath)

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
    
        
        overallMeanIou, overallMeanWeightedIou, classIoU   = get_overall_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)         
        overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)
        overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames)
#        
        print('\n\IOU Values:')
        print(model_name)
        print('GAlleryOnly Flag:', onlyGallery)
        print('The overallMeanIou = {:.3f}  '.format(overallMeanIou))
        print('The overallMeanWeightedIou = {:.3f}'.format(overallMeanWeightedIou))
        print('The overallMeanClassIou = {:.3f})'.format(overallMeanClassIou))
        print('The overallMeanWeightedClassIou = {:.3f})'.format(overallMeanWeightedClassIou))
        print('The overallMeanAvgPixAcc = {:.3f}'.format(overallMeanAvgPixAcc))
        print('The overallMeanWeightedPixAcc = {:.3f} '.format(overallMeanWeightedPixAcc))
        
        #Save results
        savefile = model_name + '_results.p'
        results = {'overallMeanIou': overallMeanIou, 'overallMeanWeightedIou': overallMeanWeightedIou, 'classIoU': classIoU, \
                    'overallMeanClassIou': overallMeanClassIou, 'overallMeanWeightedClassIou': overallMeanWeightedClassIou,  'classwiseClassIoU': classwiseClassIoU, \
                    'overallMeanAvgPixAcc': overallMeanAvgPixAcc, 'overallMeanWeightedPixAcc': overallMeanWeightedPixAcc, 'classPixAcc':classPixAcc \
                    }
        
        pickle.dump(results, open(savefile, "wb"))
        
        plot_classwiseResults(classIoU, model_name + 'classIoU' )
        plot_classwiseResults(classIoU, model_name + 'classwiseClassIoU' )
        plot_classwiseResults(classIoU, model_name + 'classPixAcc' )
        
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

       
def getBoundingBoxes(data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'):
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
    
    base_ui_path = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
    base_im_path = '/mnt/amber/scratch/Dipu/RICO/combined/'
    
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
      

def plot_classwiseResults(classwiseResult, name):
    from matplotlib import pyplot as plt
    import collections
   
    D = classwiseResult

    for k, v in D.items():
        if D[k] == []:
            D[k]= [0,0]
        else:
            D[k] = [np.mean(v), len(v)]
    D = sorted(D.items(), key=lambda kv: kv[1][0], reverse=False)
    D = collections.OrderedDict(D)
         
    
    fig, ax = plt.subplots()
    for i, (k,v) in enumerate(D.items()):
        ax.text( v[0]+0.001, i+0.25 , '{:.2f} ({})'.format(v[0], v[1]), fontsize=10,  fontweight='bold', color= 'b',  verticalalignment='top')
        print (i ,k ,v)
#    fig.title(name)
    fig.set_size_inches(7, 5) 
    ax.barh(range(len(D)), [x[0] for x in D.values()], align='center')
    plt.yticks(range(len(D)), list(D.keys()), rotation='horizontal') 
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout(h_pad=1)
    plt.show()
    plt.savefig('Results/Result_Figures/{}.png'.format(name), dpi = 500)

if __name__ == '__main__':
    main()