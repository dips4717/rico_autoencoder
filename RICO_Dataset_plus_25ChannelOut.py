#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:05:38 2019
RICO dataset for Pytorch
 + Loads images from list, 
 + Input args - paths to images, list [Train/Test]
@author: dipu
"""

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')

class RICO_Dataset(Dataset):
   
    def __init__(self, img_list, root_dir, transform=None, loader = default_loader):
        """
        Args:
            img_list (list): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_list = img_list
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.Channel_img_dir = '/mnt/amber/scratch/Dipu/RICO/25ChannelImages'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir,
                                self.img_list[index])
        im_fn = self.img_list[index][:-4]
        img = self.loader(img_name)
                
        channel25_path = os.path.join(self.Channel_img_dir, im_fn + '.npy' )
        img25Channel = np.load(channel25_path)    
        img25Channel = torch.tensor(img25Channel.astype(np.float32))

        if self.transform is not None:
            img = self.transform(img)
        return img, im_fn, img25Channel
    
       
