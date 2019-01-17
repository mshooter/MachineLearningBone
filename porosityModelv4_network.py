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

n_epochs = 50 
n_batch_size = 32
#----------------------------------------------------------------------------------------------------------------------------
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

import h5py
f = h5py.File('/home/s4906706/Documents/RaD/data/processed_data/por_Data.hdf5', 'r')

images = f['dataset_images']
labels = f['dataset_labels']

np_images = np.array(images.value)
np_labels = np.array(labels.value)

f.close()

with tf.device(assign_to_device('/gpu:0', '/cpu:0')):
    images_train, images_val, label_train, label_val = train_test_split(np_images, np_labels, test_size = 0.1)
    # should I reshape? 
    images_train = images_train.reshape(images_train.shape[0], 150,150,1).astype('float32')
    images_val= images_val.reshape(images_val.shape[0], 150,150,1).astype('float32')
    # hot encoded labels
    label_train = keras.utils.to_categorical(label_train)
    label_val = keras.utils.to_categorical(label_val)

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

model = keras.Sequential([   
    keras.layers.Conv2D(64, kernel_size=5, activation='relu',input_shape=(150,150,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
    keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dropout(0.5),
    # two outputs
    keras.layers.Dense(20, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('/home/s4906706/Desktop/accuracy_epoch_pltv4.png') 
    # summarizy history for loss
    plt.plot(historyVal['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('/home/s4906706/Desktop/loss_epoch_pltv4.png') 
plotHistory(history.history)
#----------------------------------------------------------------------------------------------------------------------------
# saves an entire model
#This single HDF5 file will contain:
#the architecture of the model (allowing the recreation of the model)
#the weights of the model
#the training configuration (e.g. loss, optimizer)
#the state of the optimizer (allows you to resume the training from exactly where you left off)
VERSION = "01"
model.save("/home/s4906706/Documents/RaD/data/por_ModelV4.hdf5")