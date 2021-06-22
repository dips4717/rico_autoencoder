#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:38:07 2019

@author: dipu
"""

import json
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import shutil
import random
import os
import pickle
import collections

global component_freq, icon_freq, textbutton_freq

def main():
    sui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
    rui_path = '/mnt/scratch/Dipu/RICO/combined/'

    samples = random.sample(range(100,2500),200)
#    samples = [100,200,888]
#    samples = [6806]
#    sample= 28970
#    shutil.copy(sui_path+'%d'%(sample)+'.json',  '/home/dipu/codes/stacked-autoencoder-pytorch/sui_temp.json')
#    shutil.copy(rui_path + '%d'%(sample) +'.json',  '/home/dipu/codes/stacked-autoencoder-pytorch/rui_temp.json' )
#    
#    with open('sui_temp.json', "r") as f:
#        sui = json.load(f)
#        
     
    visualize_uis(rui_path, sui_path, samples)
    
#    component_freq, icon_freq, textbutton_freq = count_elements(sui_path)
#    plot_bars(component_freq, icon_freq, textbutton_freq)
#    
#    return component_freq, icon_freq, textbutton_freq 

#%%    
def plot_bars(component_freq, icon_freq, textbutton_freq):
    D = component_freq
    D = sorted(D.items(), key=lambda kv: kv[1], reverse=False)
    D = collections.OrderedDict(D)
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    for i, (k,v) in enumerate(D.items()):
        ax.text( v+0.001, i+0.25 , ' {}'.format(v), fontsize=10,  fontweight='bold', color= 'b',  verticalalignment='top')
        print (i ,k ,v)
    
    ax.barh(range(len(D)), list(D.values()), align='center')
    plt.yticks(range(len(D)), list(D.keys()), rotation='horizontal')
    plt.margins(0.0)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout(h_pad=1)
    plt.show()
    plt.savefig('/mnt/scratch/Dipu/RICO/additional_annotations/component_bars.png', dpi = 500)
    
    D = icon_freq
    D = sorted(D.items(), key=lambda kv: kv[1], reverse=False)
    D = collections.OrderedDict(D)
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 40)
    for i, (k,v) in enumerate(D.items()):
        ax.text( v+0.001, i+0.25 , ' {}'.format(v), fontsize=8,   color= 'b',  verticalalignment='top')
        print (i ,k ,v)
    
    ax.barh(range(len(D)), list(D.values()), align='center')
    plt.yticks(range(len(D)), list(D.keys()), rotation='horizontal')
    plt.margins(0.0)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout(h_pad=1)
    plt.show()
    plt.savefig('/mnt/scratch/Dipu/RICO/additional_annotations/icon_bars.png', dpi = 500)
    
    
    D = textbutton_freq
    D = sorted(D.items(), key=lambda kv: kv[1], reverse=False)
    D = collections.OrderedDict(D)
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 50)
    for i, (k,v) in enumerate(D.items()):
        ax.text( v+0.001, i+0.25 , ' {}'.format(v), fontsize=8,   color= 'b',  verticalalignment='top')
        print (i ,k ,v)
    
    ax.barh(range(len(D)), list(D.values()), align='center')
    plt.yticks(range(len(D)), list(D.keys()), rotation='horizontal')
    plt.margins(0.0)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout(h_pad=1)
    plt.show()
    plt.savefig('/mnt/scratch/Dipu/RICO/additional_annotations/textbutton_bars.png', dpi = 500)
    
    
  
#%%    
def count_elements(sui_path):
    global component_freq, icon_freq, textbutton_freq 
    all_jsons = [f for f in os.listdir(sui_path) if (os.path.isfile(os.path.join(sui_path, f)) &  (os.path.splitext(f)[1] == ".json"))]
      
    component_freq = {}
    icon_freq = {}
    textbutton_freq = {}
    print ('Total files: %d'%(len(all_jsons)))
    
    for i, jsonFile in enumerate(all_jsons):
        jsonPath = os.path.join(sui_path,jsonFile)
        if (i%1000) == 0:
            print('Processed %d file'%(i))  
        
        with open(jsonPath, "r") as f:
            sui = json.load(f)
        component_freq, icon_freq, textbutton_freq  = parse_json(sui, component_freq, icon_freq, textbutton_freq)
                
    # Saving the lists and dictionaries of the component classes
    with open('/mnt/scratch/Dipu/RICO/additional_annotations/componentLabel2.txt', 'w') as f:
        for key, value in component_freq.items():
            f.write("%s %d\n"%(key,value))
            
    with open('/mnt/scratch/Dipu/RICO/additional_annotations/iconClass2.txt', 'w') as f:
        for key, value in icon_freq.items():
            f.write("%s %d\n"%(key,value))        
   
    
    with open('/mnt/scratch/Dipu/RICO/additional_annotations/textButtonClass2.txt', 'w') as f:
        for key, value in textbutton_freq.items():
            f.write("%s %d\n"%(key,value))     
    
    ComponentClasses = {'componentLabels': component_freq, 'textButtonClass': textbutton_freq, 'iconClass': icon_freq }
    pickle.dump(ComponentClasses, open("/mnt/scratch/Dipu/RICO/additional_annotations/ComponentClasses2.p", "wb"))
    
    return component_freq, icon_freq, textbutton_freq    


#%%            
def parse_json(sui, component_freq, icon_freq, textbutton_freq):
#    global component_freq, icon_freq, textbutton_freq
    for i in range(len(sui['children'])):
        
        c_name = sui['children'][i].get('componentLabel') 
        if c_name != None:
            if (c_name in component_freq):
                component_freq[c_name] +=1
            else:
                component_freq[c_name] = 1
        
        i_name = sui['children'][i].get('iconClass')
        if i_name != None:
            if (i_name in icon_freq):
                icon_freq[i_name] +=1
            else:
                icon_freq[i_name] = 1
                
                
        t_name = sui['children'][i].get('textButtonClass')
        if t_name != None:
            if (t_name in textbutton_freq):
                textbutton_freq[t_name] +=1
            else:
                textbutton_freq[t_name] = 1     
                
        # Recursion for the nested components/elements    
        if sui['children'][i].get('children') != None:
            component_freq, icon_freq, textbutton_freq  =parse_json(sui['children'][i], component_freq, icon_freq, textbutton_freq)        
    
    return component_freq, icon_freq, textbutton_freq
    
def parse_plot_ui_elements(sui, s_img, r_img, sample, ax):
    """
    Parse the json file iteratively using recursion, unwinding the all the nested children.
    Each of the component/object/bouding boxes are then plotted.
    """
    x_scale = r_img.size[0] / s_img.size[0]
    y_scale = r_img.size[1] / s_img.size[1]
    
    n_uis = len(sui['children'])
    for i in range(n_uis):
        component_Label = sui['children'][i]['componentLabel']
        
#        if component_Label == 'List Item':    # Remove this line to plot all the components (classes)      
        [x1, y1, x2, y2] = sui['children'][i]['bounds']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor= 'none', label = component_Label)
        ax[1].add_patch(rect)
        ax[1].text(x1+10, y1+10, component_Label, fontsize=8, color= 'r', verticalalignment='top')
        
        rect = patches.Rectangle((x1*x_scale, y1*y_scale), (x2-x1)*x_scale, (y2-y1)*y_scale, linewidth=2, edgecolor='r', facecolor= 'none', label = component_Label)
        ax[0].add_patch(rect)
        ax[0].text(x1*x_scale+5, y1*x_scale+5, component_Label, fontsize=6, color= 'r', verticalalignment='top')
        
        if sui['children'][i].get('children') != None:
            parse_plot_ui_elements(sui['children'][i], s_img, r_img, sample, ax)

def visualize_uis(rui_path, sui_path, samples):
    print (len(samples))
    for i, sample in enumerate(samples):
        
        r_image = rui_path + '%d'%(sample) + '.jpg'
        s_image = sui_path + '%d'%(sample) + '.png'
        if not(os.path.exists(r_image)):
            continue
        r_img = Image.open(r_image).convert('RGB')
        s_img = Image.open(s_image).convert('RGB')
        
        shutil.copy(sui_path+'%d'%(sample)+'.json',  '/home/dipu/codes/AutoEnconder_RicoDataset/sui_temp.json')
                
        with open('sui_temp.json', "r") as f:
           sui = json.load(f)                        # Semantic UIs
        
        fig, ax = plt.subplots(1,2)
        plt.setp(ax,  xticklabels=[], yticklabels=[])
        ax[1].imshow(s_img)
        ax[1].axis('off')
        ax[0].imshow(r_img)
        ax[0].axis('off') 
        print(i)
            
        parse_plot_ui_elements(sui, s_img, r_img, sample, ax)
        plt.savefig('Visualize_ui/%s'%(sample), dpi=500)
#        plt.show()
        plt.pause(0.001)
#        plt.close()

#%%
if __name__ == '__main__':
    main()
#   component_freq, icon_freq, textbutton_freq = main()