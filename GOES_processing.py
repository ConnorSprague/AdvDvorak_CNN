#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:53:08 2018
GOES preprocessing, cleaning, and augmentation

@author: d4rkmatter
"""
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils

def nested_dataset_gen(filepath,band_list,filetype, simple_read):
    """
    Takes in path to directories containing images
    seperates into specified image channels
    further seperates into file types (.tif, .jpg) specified in call
    sorts images by date/time
    creates labels for images
    returns array split into images occuring on same datetime, in specified bands
    """
    image_list = []
    
    directories = os.listdir(filepath)
    for directory in directories:
        dir_path = filepath+str(directory)
        dir_files = os.listdir(dir_path)
        for file in dir_files:
            image_path = filepath+str(directory)+'/'+file
            if not simple_read:
                if any(x in image_path for x in band_list):
                    if filetype in file:
                        holder = [image_path, int(file[-7:-4].strip('_'))]
                        image_list.append(holder)
                    else:
                        pass
                else:
                    pass
            else:
                holder = [image_path, int(file[-7:-4].strip('_'))]
                image_list.append(holder)
    images = sorted(image_list, key= lambda x: x[0])
    labels = [x[0][:-10] for x in images]
    
    images_array = np.asarray(images)
    split_images_array = np.split(images_array, len(set(labels)))
    return(split_images_array, labels)


def dataset_gen(filepath,band_list,filetype,simple_read):
    """
    reads directory containing images
    seperates into specified image channels
    further seperates into file types (.tif, .jpg) specified in call
    sorts images by date/time
    creates labels for images
    returns array split into images occuring on same datetime, in specified bands
    """
    image_list = []

    dir_files = os.listdir(filepath)
    for file in dir_files:
        image_path = filepath+'/'+file
        if not simple_read:
            if any(x in image_path for x in band_list):
                if filetype in file:
                    holder = [image_path, int(file[-7:-4].strip('_'))]
                    image_list.append(holder)
                else:
                    pass
            else:
                pass
        else:
            holder = [image_path, int(file[-7:-4].strip('_'))]
            image_list.append(holder)
    images = sorted(image_list, key= lambda x: x[0])
    labels = [x[0][:-10] for x in images]
    
    images_array = np.asarray(images)
    split_images_array = np.split(images_array, len(set(labels)))
    return(split_images_array, labels)


def image_merge(image_set,write):
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
    if write:
        """Save the needed multi channel image"""
        cv2.imwrite(filepath,needed_multi_channel_img)
        print(filepath[-23:]+'    Saved to    '+directory)
    else:
        return(needed_multi_channel_img,classname)
    

def readfiles_numpy_testtrain(filepath, test_size, random_state, classes):
    '''read in images from directory, numpy-ify,
    shuffle, split into test-train
    return x-train,test and y-train,test'''
    
    img_list = os.listdir(filepath)
    
    img_data_list=[]
    img_path_list = []
    for img in img_list:
        img_path = filepath + '/'+ img
        img_path_list.append(img_path)
        img = image.load_img(img_path, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #print('Input image shape:', x.shape)
        img_data_list.append(x)
    
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data=np.rollaxis(img_data,1,0)
    img_data=img_data[0]
    print ('Image Dataset Shape ',img_data.shape)
    
    
    # Define the number of classes
    num_classes = classes
    labels = []
    
    for image_path in img_path_list:
        class_label = image_path[-7:-4].strip('_') #get labels from filename
        class_label = (int(class_label)/5)-5 #adjust labels for one hot conversion
        labels.append(class_label)
    labels = np.asarray(labels)
    # convert class labels to one-hot encoding
    Y = np_utils.to_categorical(labels,num_classes)
    print('Labels ',Y.shape)
    #Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=random_state)
    print('Shuffled')
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    print('ready...')

    return(X_train,X_test, y_train, y_test)
