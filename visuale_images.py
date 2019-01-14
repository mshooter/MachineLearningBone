import time
import os
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras
import cv2

linux = True
n_epochs = 10
n_error = 5
#----------------------------------------------------------------------------------------------------------------------------
# get the dataset it is stored in a hdf5
import h5py 
if linux:
    f = h5py.File("/home/s4928793/RDProject/datasetFiles/data_sNs.hdf5", 'r')
else: 
    f = h5py.File("/Users/moirashooter/RDProject/datasetFiles/data_sNs.hdf5", 'r')
# get the data
with tf.device('/gpu:0'):
    images = f['dataset_images'][...]
    labels = f['dataset_labels']
    # strore them in np.arrays 
    # amount of images 
cv2.imwrite("/home/s4928793/Desktop/test.jpg", images[0][0])
# do something with data
f.close()
#----------------------------------------------------------------------------------------------------------------------------
# split into training and test set (eg 80/20)
#images_train, images_val, label_train, label_val = train_test_split(np_images, np_labels, test_size = 0.1, shuffle = True)
 # should I reshape? 
#images_train = images_train.reshape(images_train.shape[0], 150,150,1).astype('float32')
#images_val= images_val.reshape(images_val.shape[0], 150,150,1).astype('float32')
# hot encoded labels
#label_train = keras.utils.to_categorical(label_train)
#label_val = keras.utils.to_categorical(label_val)
#----------------------------------------------------------------------------------------------------------------------------
# create model
# should be adding convolutional layers
# added drop out just to be sure
model = keras.Sequential([   
    keras.layers.Conv2D(120, kernel_size=5, activation='relu',input_shape=(150,150,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=5, activation='relu'),
    keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation=tf.nn.softmax)])

