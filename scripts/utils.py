#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:42:38 2020

@author: dipu
"""


from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def visualize_sui_rui_pairs(img_id, info,
                            sui_path = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/',
                            rui_path = '/mnt/amber/scratch/Dipu/RICO/combined/'):

    
    r_image = rui_path + '%s'%(img_id) + '.jpg'
    s_image = sui_path + '%s'%(img_id) + '.png'

    r_img = Image.open(r_image).convert('RGB')
    s_img = Image.open(s_image).convert('RGB')    
    
    
    x_scale = r_img.size[0] / s_img.size[0]
    y_scale = r_img.size[1] / s_img.size[1]
    
    fig, ax = plt.subplots(1,2)
    plt.setp(ax,  xticklabels=[], yticklabels=[])
    ax[1].imshow(s_img)
    ax[1].axis('off')
    ax[0].imshow(r_img)
    ax[0].axis('off') 
    
    info_ = info[img_id]
    n_uis = info_['nComponent']
    
    for i in range(n_uis):
        component_Label = info_['componentLabel'][i]
        [x, y, w,h] = info_['xywh'][i]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor= 'none', label = component_Label)
        ax[1].add_patch(rect)
        ax[1].text(x+10, y+10, component_Label, fontsize=8, color= 'r', verticalalignment='top')
        
        rect = patches.Rectangle((x*x_scale, y*y_scale), w*x_scale, h*y_scale, linewidth=2, edgecolor='r', facecolor= 'none', label = component_Label)
        ax[0].add_patch(rect)
        ax[0].text(x*x_scale+5, y*x_scale+5, component_Label, fontsize=6, color= 'r', verticalalignment='top')
 



com2index = {
        'Toolbar':          1,
        'Image':            2,
        'Icon':             3,
        'Web View':         4,
        'Text Button':      5,
        'Text':             6,
        'Multi-Tab':        7,
        'Card':             8,
        'List Item':        9,
        'Advertisement':    10,
        'Background Image': 11,
        'Drawer':           12,
        'Input':            13,
        'Bottom Navigation':14,
        'Modal':            15,
        'Button Bar':       16,
        'Pager Indicator':  17,
        'On/Off Switch':    18,
        'Checkbox':         19,
        'Map View':         20,
        'Radio Button':     21,
        'Slider':           22,
        'Number Stepper':   23,
        'Video':            24,
        'Date Picker':      25,
        }
      
        
        