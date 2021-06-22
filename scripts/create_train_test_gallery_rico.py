import os 
import numpy as np
import pickle

from utils import visualize_sui_rui_pairs
from utils import com2index

index2com = dict((v,k) for k,v in com2index.items())

UI_data = pickle.load(open('/mnt/amber/scratch/Dipu/RICO/UI_data.p', 'rb'))
train_uis = UI_data['train_uis'] 
train_uis = [x.split('.')[0] for x in train_uis]
test_uis = UI_data['test_uis'] 
test_uis = [x.split('.')[0] for x in test_uis]

UI_test_data = pickle.load(open("/mnt/amber/scratch/Dipu/RICO/UI_test_data.p", "rb"))
query_uis = UI_test_data['query_uis']
query_uis = [x.split('.')[0] for x in query_uis]
gallery_uis = UI_test_data['gallery_uis']
gallery_uis = [x.split('.')[0] for x in gallery_uis]


#nocomp_imlist = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/no_component_imglist.pkl', 'rb'))
#train_uis = list(set(train_uis) - set(nocomp_imlist))
#gallery_uis = list(set(gallery_uis) - set(nocomp_imlist))
#query_uis = list(set(query_uis) - set(nocomp_imlist))

#%% Load the RICO annotation file and remove the unwanted images,.
info = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/rico_box_info.pkl', 'rb'))
info = dict(info)
print('size original: ', len(info), '\n')

# Remove images with no components
# Already incorporated in rico_box_info.pkl, just to make sure
nocomp_imlist = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/no_component_imglist.pkl', 'rb'))
for img_id in nocomp_imlist:
    info.pop(img_id, None)
print('info size after removing no components images', len(info), '\n')

# Remove images with number of compoenents greater than 100.
uis_ncomponent_g100 = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/ncomponents_g100_imglist.pkl', 'rb'))
for img_id in uis_ncomponent_g100:
    info.pop(img_id, None)   
print('size uis_ncomponent_g100: ', len(uis_ncomponent_g100)) 
print('size after removing uis_ncomponent_g100: ', len(info), '\n')  


#Remove the images with (horizontal) images, annotations are often not correct.
# Invalid size images:
image_size_dict = pickle.load(open('/home/dipu/codes/faster_RCNNs/RICO-simple-faster-rcnn/data/RICO_data/RICO_JPG_image_size.pickle', 'rb'))
invalid_sized_img = image_size_dict[(960,540)] + image_size_dict[(1920,1080)]  # landscape ui, probly annotation not correct.
for img_id in invalid_sized_img:
    #del info[img_id]
    info.pop(img_id,None)
print ('size of invalid_sized_img', len(invalid_sized_img))    
print('info size after removing invalid sized images: ', len(info), '\n')    


# Remove images with bounding boxes with zero width or height
# Note: bbox with negative w and h are already removed in rico_box_info.pkl
#zero_bbox_wh = [k for k, v in info.items() if (np.any(np.stack(v['xywh'])[:,2] <= 0) or np.any(np.stack(v['xywh'])[:,3] <= 0)) ]
#visualize_sui_rui_pairs(zero_bbox_wh[0], info)
    
#for i in range(20):
#visualize_sui_rui_pairs(invalid_sized_img[19], info)




#%% Count the images, number of components. 
from collections import Counter

def count_uis (img_ids):
    '''
    c1: Total instances of the uis in the list (img_ids)
    c2: Total number of images that contains the particular uis in the list(img_ids)
    '''
    c1 = np.zeros((25), dtype=int)
    c2 = np.zeros((25), dtype=int)

    for img_id in img_ids:
        temp_info = info[img_id]
        
        index = Counter(temp_info['class_id']).keys() 
        count = Counter(temp_info['class_id']).values()
        
        index = list(index)
        index = [x-1 for x in index]
        count = list(count)
        
        temp_count = np.zeros((25), dtype=int)
        temp_count[index] = count
        c1 += temp_count
        c2 += temp_count>0     
    return c1, c2
        


#%% 
'''Divide the dataset into train gallery, test
Initialize from the previous train test gal set'''

# Remove the images that are not in RICO info dictionary.
print ('Length Train, Gallery, Query: ', len(train_uis), len(gallery_uis), len(query_uis))
train_uis = list(set(train_uis) & set(info.keys()))
gallery_uis = list(set(gallery_uis) & set(info.keys()))
query_uis = list(set(query_uis) & set(info.keys()))
print ('Length Train, Gallery, Query: ', len(train_uis), len(gallery_uis), len(query_uis))


# Count the current numbers. c1 = total num of instances. c2 = num of images that contain the uis.
c1, c2 = count_uis(info.keys())
c1_t, c2_t = count_uis(train_uis)
c1_q, c2_q = count_uis(query_uis) 
c1_g, c2_g = count_uis(gallery_uis)     

c1_all = np.vstack((c1,c1_t,c1_g,c1_q)).transpose()  # Concatenate all the values for easier visualization of data
c2_all = np.vstack((c2,c2_t,c2_g,c2_q)).transpose()  

# Work using c2 metric , ie, total number of images that contain the uis than total instances (c1)
# Iterate through all the query images:
import random


sum_tq_ids = len(train_uis) + len(query_uis)
for ii in range(c2_t.shape[0]):
    if c2_q[ii] < 5:
        n_toAdd = 5 - c2_q[ii]
        
        class_id = ii +1    # class indexing in rico info starts from 1
        class_img_ids = [k for k, v in info.items() if class_id in v['class_id'] ]
        pool = list (set(class_img_ids) & set(train_uis))  # creat possible pool of ids from train uis
        ids_toAdd = random.sample(pool, n_toAdd)
        train_uis = [x for x in train_uis if x not in ids_toAdd]
        query_uis = query_uis + ids_toAdd
        
        print ('%d images of class %s added to query images from train'%(n_toAdd, index2com[class_id]))
        assert (len(train_uis)+len(query_uis) == sum_tq_ids)
            
c1_t2, c2_t2 = count_uis(train_uis)
c1_q2, c2_q2 = count_uis(query_uis) 
c1_g2, c2_g2 = count_uis(gallery_uis)       

c1_all2 = np.vstack((c1,c1_t2,c1_g2,c1_q2)).transpose()  # Concatenate all the values for easier visualization of data
c2_all2 = np.vstack((c2,c2_t2,c2_g2,c2_q2)).transpose()        
            
print ('Length Train, Gallery, Query: ', len(train_uis), len(gallery_uis), len(query_uis))        


# Save this to the disk..
import pickle
rico_split_sets = {'train_uis': train_uis, 'gallery_uis': gallery_uis, 'query_uis': query_uis}
save_file = "/mnt/amber/scratch/Dipu/RICO/rico_split_set2.pkl"
pickle.dump(rico_split_sets, open(save_file, "wb"))
print('\n', 'saved the split set into ', save_file )
 



#%% Visualize sample images from a fix category 
import shutil
import os 
import random


cat = 8
cat_ids = [k for k, v in info.items() if cat in v['class_id'] ]

save_dir = 'classwise_images/%s/'%index2com[cat]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sel_ids = random.sample(cat_ids,50)

for ids in sel_ids:
    shutil.copy('/mnt/amber/scratch/Dipu/RICO/combined/' + ids + '.jpg', save_dir) 

for i in range(20):
    visualize_sui_rui_pairs(cat_ids[i], info)

#%% Output
#    
#size original:  65538 
#
#info size after removing no components images 65538 
#
#size uis_ncomponent_g100:  217
#size after removing uis_ncomponent_g100:  65321 
#
#size of invalid_sized_img 67
#info size after removing invalid sized images:  65261 
#
#Length Train, Gallery, Query:  53008 13203 50
#Length Train, Gallery, Query:  52183 13028 50
#2 images of class Multi-Tab added to query images from train
#3 images of class Card added to query images from train
#1 images of class Background Image added to query images from train
#3 images of class Drawer added to query images from train
#5 images of class Bottom Navigation added to query images from train
#1 images of class Modal added to query images from train
#5 images of class Button Bar added to query images from train
#4 images of class Pager Indicator added to query images from train
#5 images of class On/Off Switch added to query images from train
#5 images of class Checkbox added to query images from train
#5 images of class Map View added to query images from train
#4 images of class Radio Button added to query images from train
#3 images of class Slider added to query images from train
#5 images of class Number Stepper added to query images from train
#5 images of class Video added to query images from train
#5 images of class Date Picker added to query images from train
#Length Train, Gallery, Query:  52122 13028 111    



     
    
    







