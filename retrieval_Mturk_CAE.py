#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:26:12 2020

@author: dipu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:16:53 2020

@author: dipu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:01:07 2019
same as perform_tests.py
but removes the images with no components and images with number of components greater than 100
@author: dipu
"""
import os 
import torch 
from torchvision import transforms 
import glob
import json
from collections import defaultdict
#from utils import extract_features, compute_iou
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
from RICO_Dataset import RICO_Dataset
from models.model_upsample_emb512 import CAE_upsample_dim512 
from utils import mkdir_if_missing, load_checkpoint


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def extract_features(data_loader, model):
    model.eval()
    torch.set_grad_enabled(False)
    features = []
    labels = []
   
    for i, (imgs, im_fn) in enumerate(data_loader):
    #for i, (imgs, im_fn, img25Chan) in enumerate(data_loader):    
        imgs = imgs.cuda()
        x_enc = model(imgs, training=False)
        outputs = x_enc.detach().cpu().numpy()
        features.append(outputs)
        labels += list(im_fn)
#        print(i)    
    return features, labels  

def remove_null_and_large_comp_images(img_list):
    nocomp_imlist = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/no_component_imglist.pkl', 'rb'))
    ncomp_g100_imglist = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/ncomponents_g100_imglist.pkl', 'rb'))
    nocomp_imlist = [x +'.png' for x in nocomp_imlist]
    ncomp_g100_imglist = [x + '.png' for x in ncomp_g100_imglist]
    
    img_list = list(set(img_list) - set(nocomp_imlist))
    img_list = list(set(img_list) - set(ncomp_g100_imglist))
    return img_list


data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
UI_data = pickle.load(open("/mnt/amber/scratch/Dipu/RICO/UI_data.p", "rb"))
UI_test_data = pickle.load(open("/mnt/amber/scratch/Dipu/RICO/UI_test_data.p", "rb"))
train_uis = UI_data['train_uis']
query_uis = UI_test_data['query_uis']
gallery_uis = UI_test_data['gallery_uis']


train_uis = remove_null_and_large_comp_images(train_uis)
query_uis = remove_null_and_large_comp_images(query_uis)
gallery_uis = remove_null_and_large_comp_images(gallery_uis)


data_transform = transforms.Compose([
        transforms.Resize([255,127]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])   

BATCH_SIZE = 128

query_dataset = RICO_Dataset(query_uis, data_dir, transform= data_transform)
query_loader = torch.utils.data.DataLoader(query_dataset, batch_size= BATCH_SIZE, shuffle=False,
                                           drop_last = False, pin_memory=True, num_workers=16)

gallery_dataset = RICO_Dataset(gallery_uis, data_dir, transform= data_transform)
gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size= BATCH_SIZE, shuffle=False,
                                           drop_last = False, pin_memory=True, num_workers=16)


   

def main():     
    
    boundingBoxes = getBoundingBoxes()
    model_name = 'model_CAE_emb512'
    onlyGallery = True
    model = CAE_upsample_dim512()
    model_path = '/home/dipu/codes/AutoEnconder_RicoDataset/runs/{}/ckp_ep20.pth.tar'.format(model_name)
    resume = load_checkpoint(model_path)
    model.load_state_dict(resume['state_dict'])
    model = model.cuda()
    model.eval()
    
    q_feat, q_fnames = extract_features(query_loader, model)
    g_feat, g_fnames = extract_features(gallery_loader, model)
    print('extracted features from {} query images'.format(len(q_fnames)))
    print('extracted features from {} gallery images'.format(len(g_fnames)))
    #t_feat, t_fnames = extract_features(train_loader, model)
    
    
    q_feat = np.concatenate(q_feat)
    g_feat = np.concatenate(g_feat)

    distances = cdist(q_feat, g_feat, metric= 'euclidean')
    sort_inds = np.argsort(distances)
          
    overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
    overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     

    print(model_name)
    print('GAlleryOnly Flag:', onlyGallery)        
    print('The overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
    print('The overallMeanWeightedClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedClassIou]) + '\n')
    print('The overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
    print('The overallMeanWeightedPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedPixAcc]) + '\n')
        
#%%    
    import shutil
    base_img = 'ConvAutoEncoder_Images/'
    base_sui = 'ConvAutoEncoder_Semantic_UIs/'

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

#%% ploting
    plot_retrieved_images_and_uis(sort_inds, q_fnames, g_fnames, model_name)
    



#%%  
from matplotlib import pyplot as plt            
def plot_retrieved_images_and_uis(sort_inds, query_uis, gallery_uis, model_name):
    base_im_path = '/mnt/amber/scratch/Dipu/RICO/combined/'
    base_ui_path = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
    
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
            
        directory =  'Retrieved_Images_CAE/'
        if not os.path.exists(directory):
            os.makedirs(directory)  
            
        plt.savefig( directory + str(i) + '.png')
       # plt.pause(0.1)
        plt.close()
        #print('Wait')
        print(i)



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



if __name__ == '__main__':
    main()