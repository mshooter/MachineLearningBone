###-------------------------------------------------------------------------------------------------------
## author: Moira Shooter
## fileName: evaluate_test.py 
## brief: This file was used to evaluate and test the predictions of the neural network 
## also to plot the results
###-------------------------------------------------------------------------------------------------------
import os 
import cv2 as cv
import time
import h5py 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
#-----------------------------------------------------------------------------------------------------------
# my variables
#-----------------------------------------------------------------------------------------------------------
isLinux = True    
class_names = ['health', 'sick']
#-----------------------------------------------------------------------------------------------------------
### The following section is from :-
### tfboyd(2017). Placing Variables on the cpu using `tf.contrib.layers` functions [online]. [Accesses 2018]
### Available from: "https://github.com/tensorflow/tensorflow/issues/9517".
#-----------------------------------------------------------------------------------------------------------
PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign
### end of Citation
#-----------------------------------------------------------------------------------------------------------
### The following section is from :-
### Machine Learning Guru. Saving and loading a large number of images (data) into a single HDF5 file[online]. [Accesses 2018]
### Available from: "http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html".
#-----------------------------------------------------------------------------------------------------------
if isLinux:
    model = keras.models.load_model("/home/s4928793/Desktop/RDFinal/models/v03_model_health.hdf5")
    f = h5py.File("/home/s4928793/RDProject/datasetFiles/data_sNs.hdf5", 'r')
else: 
    model = keras.models.load_model("/Users/moirashooter/RDProject/modelsFinal/v03_model.hdf5")
    f = h5py.File("/Users/moirashooter/RDProject/datasetFiles/data_sNs.hdf5", 'r')
# get the data
images = f['dataset_images']
labels = f['dataset_labels']
# strore them in np.arrays 
np_images = np.array(images.value)
np_labels = np.array(labels.value)
f.close()
### End of Citation
#-----------------------------------------------------------------------------------------------------------
# split into training and test set (eg 80/20)
# use gpu with linux 
#-----------------------------------------------------------------------------------------------------------
with tf.device(assign_to_device('/gpu:0','/cpu:0')):
    images_train, images_val, label_train, label_val = train_test_split(np_images, np_labels, test_size = 0.1)
    # should I reshape? 
    images_train = images_train.reshape(images_train.shape[0], 150,150,1).astype('int8')
    images_val= images_val.reshape(images_val.shape[0], 150,150,1).astype('int8')
    # hot encoded labels
    label_train = keras.utils.to_categorical(label_train)
    #label_val = keras.utils.to_categorical(label_val)
# do not need to compile when predicting
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
#-----------------------------------------------------------------------------------------------------------
# evaluation function
#-----------------------------------------------------------------------------------------------------------
def evaluate():
    evaluation_loss, evaluation_acc = model.evaluate(images_val, label_val)
    print('Test accuracy: ', evaluation_acc)
    print('Test Loss: ', evaluation_loss)
#-----------------------------------------------------------------------------------------------------------
# print predict
#-----------------------------------------------------------------------------------------------------------
with tf.device('/gpu:0'):
    predict = model.predict(images_val)
    # need to reshape to have a nice output to show 
    images_val = images_val.reshape(images_val.shape[0], 1,150,150).astype('int8')
    #-----------------------------------------------------------------------------------------------------------
    ### The following section is from :-
    ### Tensorflow(2018). Train your first neural network: basic classification [online]. [Accesses 2018]
    ### Available from: "https://www.tensorflow.org/tutorials/keras/basic_classification".
    #-----------------------------------------------------------------------------------------------------------
    r = 5
    c = 3
    num_img = r*c
    plt.figure(figsize=(2*2*c, 2*r))
    for i in range(num_img):
        # if healthy
        plt.subplot(r,2*c,2*i+1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images_val[i][0],cmap=plt.cm.binary)
        predicted_label = np.argmax(predict[i])
        if predicted_label == label_val[i]:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                         100*np.max(predict[i]),
                         class_names[label_val[i]]),
                         color=color)
    ### end of Citation
    plt.savefig('/home/s4928793/Desktop/fig1.jpg')
