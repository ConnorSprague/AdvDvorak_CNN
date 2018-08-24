
import numpy as np
import os
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
#from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf

#img_path = 'Image_Chips/AL012018_Alberto/AL012018_Alberto_201805251200_CMI_C09_35.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#print (x.shape)
#x = np.expand_dims(x, axis=0)
#print (x.shape)
#x = preprocess_input(x)
#print('Input image shape:', x.shape)

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/combined_images_4'
img_list = os.listdir(data_path)

img_data_list=[]
img_path_list = []
for img in img_list:
    img_path = data_path + '/'+ img
    img_path_list.append(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
	#x = x/255
    print('Input image shape:', x.shape)
    img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 28
num_of_samples = img_data.shape[0]
labels = []
for image_path in img_path_list:
    class_label = image_path[-7:-4].strip('_') #get labels from filename
    class_label = (int(class_label)/5)-5 #adjust labels for one hot conversion
    labels.append(class_label)
labels = np.asarray(labels)
# convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels,num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#########################################################################################
batch_size = 64
epochs = 25
with tf.device('/cpu:0'):    
    # Custom_vgg_model_1
    #Training the classifier alone
    image_input = Input(shape=(224, 224, 3))
    
    model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
    model.summary()
    last_layer = model.get_layer('fc2').output
    #x= Flatten(name='flatten')(last_layer)
    out = Dense(num_classes, activation='softmax', name='output')(last_layer)
    custom_vgg_model = Model(image_input, out)
    custom_vgg_model.summary()
    
    for layer in custom_vgg_model.layers[:-1]:
        layer.trainable = False
    
    custom_vgg_model.layers[3].trainable
    
    custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    
    t=time.time()
    
    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    
    # fits the model on batches with real-time data augmentation:
    hist = custom_vgg_model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs, verbose=1, shuffle=True)


    custom_vgg_model.save_weights('first_try.h5')  # always save your weights after training or during training
    score = custom_vgg_model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


####################################################################################################################
with tf.device('/cpu:0'): 
    #Training the feature extraction also
    
    image_input = Input(shape=(224, 224, 3))
    
    model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
    
    model.summary()
    
    last_layer = model.get_layer('block5_pool').output
    x= Flatten(name='flatten')(last_layer)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    custom_vgg_model2 = Model(image_input, out)
    custom_vgg_model2.summary()
    
    # freeze all the layers except the dense layers
    for layer in custom_vgg_model2.layers[:-3]:
        	layer.trainable = False
    
    custom_vgg_model2.summary()
    
    custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    
    t=time.time()
    #	t = now()
    hist = custom_vgg_model2.fit_generator(datagen.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train) / batch_size, epochs=12, 
                                           verbose=1, validation_data=(X_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)
    
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#%%
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
