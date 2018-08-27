#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 09:51:25 2018
Custom Keras-TF CNN for TC intensity classification

@author: conno
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import GOES_processing as gp
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import Tensorboard
from time import time

######################  Processing --> GOES_processing.py
filepath = 'combined_images_3'
test_size = 0.2
random_state = 2
num_classes = 28
x_train, x_test, y_train, y_test = gp.readfiles_numpy_testtrain(filepath,test_size,random_state, num_classes)


#######################
batch_size = 64
epochs = 40

# input image dimensions
img_rows, img_cols = 300, 300

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data() 


input_shape = (img_rows, img_cols, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples\n')


with tf.device('/cpu:0'):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    

    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    channel_shift_range=2.0)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train, callbacks = [tensorboard])
    
    # fits the model on batches with real-time data augmentation:
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs, verbose=1, shuffle=True)


    model.save_weights('logs/first_try.h5')  # always save your weights after training or during training
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
