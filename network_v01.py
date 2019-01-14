import time
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras
import cv2

isLinux = True  
n_epochs = 5 
n_batch_size = 200
#----------------------------------------------------------------------------------------------------------------------------
PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]
    
# see https://github.com/tensorflow/tensorflow/issues/9517
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
#----------------------------------------------------------------------------------------------------------------------------
# get the dataset it is stored in a hdf5
import h5py
if isLinux:
    f = h5py.File("/home/s4928793/RDProject/datasetFiles/data_sNs.hdf5", 'r')
else: 
    f = h5py.File("/Users/moirashooter/RDProject/datasetFiles/data_sNs.hdf5", 'r')
# get the data
images = f['dataset_images']
labels = f['dataset_labels']
# strore them in np.arrays 

np_images = np.array(images.value)
np_labels = np.array(labels.value)
# do something with data
f.close()
#----------------------------------------------------------------------------------------------------------------------------
# split into training and test set (eg 80/20)
# use gpu with linux 
with tf.device(assign_to_device('/gpu:0','/cpu:0')):
    images_train, images_val, label_train, label_val = train_test_split(np_images, np_labels, test_size = 0.1)
    # should I reshape? 
    images_train = images_train.reshape(images_train.shape[0], 150,150,1).astype('float32')
    images_val= images_val.reshape(images_val.shape[0], 150,150,1).astype('float32')
    # hot encoded labels
    label_train = keras.utils.to_categorical(label_train)
    label_val = keras.utils.to_categorical(label_val)
#----------------------------------------------------------------------------------------------------------------------------
# show the images to train
def showTrained():
    plt.figure(figsize=(10,10))
    for i in range(5):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_train[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[label_train[i]])
    plt.show()
#----------------------------------------------------------------------------------------------------------------------------
# create model
# should be adding convolutional layers
# added drop out just to be sure
model = keras.Sequential([   
    keras.layers.Conv2D(64, kernel_size=5, activation='relu',input_shape=(150,150,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    # two outputs
    keras.layers.Dense(2, activation=tf.nn.softmax)])
#----------------------------------------------------------------------------------------------------------------------------
# loss function - accuracy - minimize it  
# optimizer - model is updated based on data and loss 
# metrics - monitor training and test steps 
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# print model informatioon 
model.summary()
# early stop call back 
earlystop = keras.callbacks.EarlyStopping(monitor='acc', min_delta = 0.0001, patience=5, verbose =1, mode='auto') 
callbacks_list = [earlystop]
#----------------------------------------------------------------------------------------------------------------------------
# train - feed - labels and images 
# model learns to associate 
# model to make predications about a test set (test_images) - test labbels 
start = time.time()
with tf.device('/gpu:0'):
    history = model.fit(images_train, label_train, epochs=n_epochs,callbacks=callbacks_list, batch_size = n_batch_size)
end = time.time() 
print("model took %0.2f seconds to train" % (end-start))
print(history.history.keys())
def plotHistory(historyVal):
    # summarize history for accuraxy
    plt.plot(historyVal['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train'], loc='upper left')
    plt.savefig('/home/s4928793/Desktop/accuracy_epoch_plt.png') 
    # summarizy history for loss
    plt.plot(historyVal['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'loss'], loc='upper left')
    plt.savefig('/home/s4928793/Desktop/loss_epoch_plt.png') 
plotHistory(history.history)
#----------------------------------------------------------------------------------------------------------------------------
if isLinux:
    model.save("/home/s4928793/RDProject/modelsFinal/v03_model.hdf5")
    model.save_weights("/home/s4928793/RDProject/modelsFinal/v03_weights.hdf5")
else:
    model.save("/Users/moirashooter/RDProject/modelsFinal/v01_model.hdf5")
