#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:01:06 2019

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
#boundingBoxes = getBoundingBoxes()

#def main():
#    
    
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

overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,gnames,q_fnames, topk = [1,5,10])
overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,gnames,q_fnames, topk = [1,5,10])     
    
print('The overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
print('The overallMeanWeightedClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedClassIou]) + '\n')
print('The overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
print('The overallMeanWeightedPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedPixAcc]) + '\n')
        
#   
#    overallMeanIou, overallMeanWeightedIou, _   = get_overall_IOU(boundingBoxes,sort_inds,gnames,q_fnames)         
#    overallMeanClassIou, overallMeanWeightedClassIou,_ = get_overall_Classwise_IOU(boundingBoxes,sort_inds,gnames,q_fnames)
#    overallMeanAvgPixAcc, overallMeanWeightedPixAcc,_ = get_overall_pix_acc(boundingBoxes,sort_inds,gnames,q_fnames)
#    
#    print('The baseline model using 64dim emb provided, UIST 2017')
#    print('\n\n Mean IOU/PixelAcc Values:')
#    print('The overallMeanIou = {:.3f}  '.format(overallMeanIou))
#    print('The overallMeanWeightedIou = {:.3f}'.format(overallMeanWeightedIou))
#    print('The overallMeanClassIou = {:.3f})'.format(overallMeanClassIou))
#    print('The overallMeanWeightedClassIou = {:.3f})'.format(overallMeanWeightedClassIou))
#    print('The overallMeanAvgPixAcc = {:.3f}'.format(overallMeanAvgPixAcc))
#    print('The overallMeanWeightedPixAcc = {:.3f} '.format(overallMeanWeightedPixAcc))

#if __name__ == '__main__':
#    main()


