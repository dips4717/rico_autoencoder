#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:42:32 2020

@author: dipu
"""

ui_names = [f for f in os.listdir(data_dir) if (os.path.isfile(os.path.join(data_dir, f)) &  (os.path.splitext(f)[1] == ".png"))]
    random_seed = 42
    
    dataset_size = len(ui_names)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    split = int(np.floor(0.8 * dataset_size))
    train_indices, test_indices =  indices[:split] , indices[split:]
    train_uis = [ui_names[x] for x in train_indices]
    test_uis = [ui_names[x] for x in test_indices]
    
    UI_data = {"ui_names": ui_names, "train_uis" : train_uis, "test_uis": test_uis}
    pickle.dump(UI_data, open("/mnt/scratch/Dipu/RICO/UI_data.