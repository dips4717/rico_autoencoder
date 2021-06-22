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
#from models.model_CAE_emb512 import ConvAutoEncoder
from models.model_upsample_emb512 import CAE_upsample_dim512 
#from models.model_CAE_emb512_25ChannelOut import ConvAutoEncoder
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

train_dataset = RICO_Dataset(train_uis, data_dir, transform= data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= BATCH_SIZE, 
                                           drop_last = True, pin_memory=True, num_workers=16)

query_dataset = RICO_Dataset(query_uis, data_dir, transform= data_transform)
query_loader = torch.utils.data.DataLoader(query_dataset, batch_size= BATCH_SIZE, shuffle=False,
                                           drop_last = False, pin_memory=True, num_workers=16)

gallery_dataset = RICO_Dataset(gallery_uis, data_dir, transform= data_transform)
gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size= BATCH_SIZE, shuffle=False,
                                           drop_last = False, pin_memory=True, num_workers=16)


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





def main():
    boundingBoxes = getBoundingBoxes()
    
    #models = ['model_CAE_emb2688', 'model_CAE_emb512', 'model_CAE2_OnlyConv_emb2912']
    models = ['model_CAE_emb512'] # ['modelCAE_emb512_25Channel_out']
    for model_name in models:
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
        #t_feat = np.concatenate(t_feat)
        
        if not(onlyGallery):
            g_feat = np.vstack((g_feat,t_feat))
            g_fnames = g_fnames + t_fnames 

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
        
        
                
#        overallMeanIou, overallMeanWeightedIou, classIoU   = get_overall_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)         
#        overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames)
#        overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames)
##        
#        print('\n\IOU Values:')
#        print(model_name)
#        print('GAlleryOnly Flag:', onlyGallery)
#        print('The overallMeanIou = {:.3f}  '.format(overallMeanIou))
#        print('The overallMeanWeightedIou = {:.3f}'.format(overallMeanWeightedIou))
#        print('The overallMeanClassIou = {:.3f})'.format(overallMeanClassIou))
#        print('The overallMeanWeightedClassIou = {:.3f})'.format(overallMeanWeightedClassIou))
#        print('The overallMeanAvgPixAcc = {:.3f}'.format(overallMeanAvgPixAcc))
#        print('The overallMeanWeightedPixAcc = {:.3f} '.format(overallMeanWeightedPixAcc))
#        
#        #Save results
#        savefile = model_name + '_results.p'
#        results = {'overallMeanIou': overallMeanIou, 'overallMeanWeightedIou': overallMeanWeightedIou, 'classIoU': classIoU, \
#                    'overallMeanClassIou': overallMeanClassIou, 'overallMeanWeightedClassIou': overallMeanWeightedClassIou,  'classwiseClassIoU': classwiseClassIoU, \
#                    'overallMeanAvgPixAcc': overallMeanAvgPixAcc, 'overallMeanWeightedPixAcc': overallMeanWeightedPixAcc, 'classPixAcc':classPixAcc \
#                    }
#        
#        pickle.dump(results, open(savefile, "wb"))
#        
#        plot_classwiseResults(classIoU, model_name + 'classIoU' )
#        plot_classwiseResults(classwiseClassIoU, model_name + 'classwiseClassIoU' )
#        plot_classwiseResults(classPixAcc, model_name + 'classPixAcc' )
        
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
#        print (i ,k ,v)
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