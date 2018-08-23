#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:53:08 2018
GOES preprocessing, cleaning, and augmentation

@author: d4rkmatter
"""
import pandas as pd
import os
import numpy as np
from keras.preprocessing import image
import cv2

filepath = 'Image_Chips/'
band_list = ['_C09_', '_C12_','_C15_']

def dataset_gen(filepath,band_list):
    """
    Takes in path to directories containing images
    reads each directory and subsequent images
    seperates into specified image channels
    further seperates into .tif and .jpg
    returns list of .tif and .jpg iamges
    """
    jpg_list = []
    tif_list = []
    
    directories = os.listdir(filepath)
    for directory in directories:
        dir_path = filepath+str(directory)
        dir_files = os.listdir(dir_path)
        for file in dir_files:
            image_path = filepath+str(directory)+'/'+file
            #img = image.load_img(image_path)
            if any(x in image_path for x in band_list):
                if '.jpg' in file:
                    #jpg_list[0].append(image.load_img(image_path,target_size=(224,224)))
                    holder = [image_path, int(file[-7:-4].strip('_'))]
                    jpg_list.append(holder)
                else:
                    holder = [image_path, int(file[-7:-4].strip('_'))]
                    tif_list.append(holder)
            else:
                pass
    return(jpg_list, tif_list)

jpg_images, tif_images = dataset_gen(filepath,band_list)
jpg_images = sorted(jpg_images, key= lambda x: x[0])
tif_images = sorted(tif_images, key= lambda x: x[0])

class_list = [x[0][:-10] for x in jpg_images]

jpg_images_array = np.asarray(jpg_images)
split_jpg_images_array = np.split(jpg_images_array, len(set(class_list)))

def image_merge(image_set):
    """Read each channel into a numpy array. 
    Save files in filepath with datetime and class in filename
    """

    a = cv2.imread(image_set[0][0], 0)
    b = cv2.imread(image_set[1][0], 0)
    c = cv2.imread(image_set[2][0], 0)
    
    """Create a blank image that has three channels 
    and the same number of pixels as your original input"""
    needed_multi_channel_img = np.zeros((a.shape[0], a.shape[1], 3))
    
    """Add the channels to the needed image one by one"""
    needed_multi_channel_img [:,:,0] = a
    needed_multi_channel_img [:,:,1] = b
    needed_multi_channel_img [:,:,2] = c
    
    directory = 'combined_images_3/'
    imagename = image_set[0][0][-27:-10]
    classname = image_set[0][0][-7:-4].strip('_')+'.jpg'
    if imagename[0] != '2':
        imagename = '2'+imagename[:-1]
    filepath = str(directory+imagename+classname)
    """Save the needed multi channel image"""
    cv2.imwrite(filepath,needed_multi_channel_img)
    print(filepath[-23:]+'    Saved to    '+directory)
    return()
    
    
for image_set in split_jpg_images_array:
    image_merge(image_set)
print('Complete')