#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:01:07 2019
similar to perform_test_2.py
for the new rico split sets..
add argparse...

@author: dipu
"""
import os 
import torch 
from torchvision import transforms 
import glob
import json
from collections import defaultdict
from PIL import Image
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import argparse

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes


from eval_metrics.get_overall_IOU import get_overall_IOU
from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc

from RICO_Dataset import RICO_Dataset
import models
from utils import mkdir_if_missing, load_checkpoint


def extract_features(data_loader, model):
    model.eval()
    torch.set_grad_enabled(False)
    features = []
    labels = []
   
    for i, (imgs, im_fn) in enumerate(data_loader):
    #for i, (imgs, im_fn, img25Chan) in enumerate(data_loader):    
        imgs = imgs.cuda()
        x_enc, out = model(imgs)
        outputs = x_enc.detach().cpu().numpy()
        features.append(outputs)
        labels += list(im_fn)
#        print(i)    
    return features, labels  

data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations'
split_set_file = '/mnt/amber/scratch/Dipu/RICO/rico_split_set2.pkl'
rico_split_set2 = pickle.load(open(split_set_file, 'rb'))

train_uis = rico_split_set2['train_uis']
query_uis = rico_split_set2['query_uis']
gallery_uis = rico_split_set2['gallery_uis']

train_uis = [x + '.png' for x in train_uis]
query_uis = [x + '.png' for x in query_uis]  
gallery_uis = [x + '.png' for x in gallery_uis]

def main(args):
    BATCH_SIZE = args.batch_size
    
    if args.model_name == 'strided_512' or args.model_name == 'strided':
        resize_shape = [255,127]
    else:
        resize_shape = [254,126]
        
    data_transform = transforms.Compose([
            transforms.Resize(resize_shape),  #transforms.Resize([255,127])  # transforms.Resize([254,126])
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
                                                 drop_last = False, pin_memory=True, num_workers=16)
    
    #Create boundingboxes class instance and intialize with all the bboxes in rico
    boundingBoxes = getBoundingBoxes()
    
    #Model
    model = models.create(args.model_name)
    model_path = 'runs_new_rico_splitset/%s/ckp_ep20.pth.tar'%(args.model_name)
    resume = load_checkpoint(model_path)
    model.load_state_dict(resume['state_dict'])
    model = model.cuda()
    model.eval()
    
    onlyGallery = True
    q_feat, q_fnames = extract_features(query_loader, model)
    g_feat, g_fnames = extract_features(gallery_loader, model)

    #t_feat, t_fnames = extract_features(train_loader, model)
    
    
    q_feat = np.concatenate(q_feat)
    g_feat = np.concatenate(g_feat)
    #t_feat = np.concatenate(t_feat)
    
    print('extracted features from query images with shape {}'.format( q_feat.shape))
    print('extracted features from gallery images with shape {}'.format(g_feat.shape))
    
    if not(onlyGallery):
        g_feat = np.vstack((g_feat,t_feat))
        g_fnames = g_fnames + t_fnames 

    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)
    
    overallMeanIou, overallMeanWeightedIou, classIoU   = get_overall_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)         
    overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)
    overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames)
#        
    print('\n\IOU Values:')
    print(args.model_name)
    print('GAlleryOnly Flag:', onlyGallery)
    print('The overallMeanIou = {:.3f}  '.format(overallMeanIou))
    print('The overallMeanWeightedIou = {:.3f}'.format(overallMeanWeightedIou))
    print('The overallMeanClassIou = {:.3f})'.format(overallMeanClassIou))
    print('The overallMeanWeightedClassIou = {:.3f})'.format(overallMeanWeightedClassIou))
    print('The overallMeanAvgPixAcc = {:.3f}'.format(overallMeanAvgPixAcc))
    print('The overallMeanWeightedPixAcc = {:.3f} '.format(overallMeanWeightedPixAcc))
    
    #Save results in txt file 
    save_dir = 'runs_new_rico_splitset/%s/'%(args.model_name)
    savefile = save_dir + args.model_name + '.txt'
    
    with open(savefile, 'w') as f:
        f.write('GAlleryOnly Flag: {}\n'.format(onlyGallery))
        f.write('The overallMeanIou = {:.3f}  \n'.format(overallMeanIou))
        f.write('The overallMeanWeightedIou = {:.3f} \n'.format(overallMeanWeightedIou))
        f.write('The overallMeanClassIou = {:.3f})\n'.format(overallMeanClassIou))
        f.write('The overallMeanWeightedClassIou = {:.3f})\n'.format(overallMeanWeightedClassIou))
        f.write('The overallMeanAvgPixAcc = {:.3f}\n'.format(overallMeanAvgPixAcc))
        f.write('The overallMeanWeightedPixAcc = {:.3f} \n'.format(overallMeanWeightedPixAcc))
    
    # Save the all  the classwise values into a pickle file
    savefile = save_dir + args.model_name + '_results.pickle'
    results = {'overallMeanIou': overallMeanIou, 'overallMeanWeightedIou': overallMeanWeightedIou, 'classIoU': classIoU, \
                'overallMeanClassIou': overallMeanClassIou, 'overallMeanWeightedClassIou': overallMeanWeightedClassIou,  'classwiseClassIoU': classwiseClassIoU, \
                'overallMeanAvgPixAcc': overallMeanAvgPixAcc, 'overallMeanWeightedPixAcc': overallMeanWeightedPixAcc, 'classPixAcc':classPixAcc \
                }
    pickle.dump(results, open(savefile, "wb"))
    print("Saved the results to {} file".format(savefile))
    
    plot_classwiseResults(save_dir, classIoU, args.model_name + 'classIoU' )
    plot_classwiseResults(save_dir, classwiseClassIoU, args.model_name + 'classwiseClassIoU' )
    plot_classwiseResults(save_dir, classPixAcc, args.model_name + 'classPixAcc' )
        
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

       
def getBoundingBoxes(data_dir = '/mnt/scratch/Dipu/RICO/semantic_annotations/'):
    allBoundingBoxes = BoundingBoxes()
    
    files = glob.glob(data_dir+ "*.json")
    for file in files:
        imageName = os.path.split(file)[1]
        imageName = imageName.replace(".json", "")
#        print(imageName)
        
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
    print('Collected {} bounding boxes from {} images'. format(allBoundingBoxes.count(), len(files) ))        
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
      

def plot_classwiseResults(save_dir, classwiseResult, name):
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
#        print (i ,k ,v)
#    fig.title(name)
    fig.set_size_inches(7, 5) 
    ax.barh(range(len(D)), [x[0] for x in D.values()], align='center')
    plt.yticks(range(len(D)), list(D.keys()), rotation='horizontal') 
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout(h_pad=1)
    plt.show()
    plt.savefig('{}/{}.png'.format(save_dir, name), dpi = 500)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # optimization
    parser.add_argument('--batch_size', default = 128, type=int, metavar='N',  
                        help='mini-batch size (1 = pure stochastic) Default: 256') 
    # model
    parser.add_argument('--model_name', default = 'upsample_512', type = str,
                        help = 'which CNN autoencoder: upsample or strided or strided_512 or upsample_512')
    
    parser.add_argument('--gpu_id', type=str, default = '3', help = 'GPU ID')

    
    args = parser.parse_args()
    
    main(args)