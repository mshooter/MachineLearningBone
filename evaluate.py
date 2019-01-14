import os 
import time
import h5py 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
n_epochs = 5 
n_batch_size = 32
PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]
isLinux = True    
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
 
if isLinux:
    model = keras.models.load_model("/home/s4928793/RDProject/modelsFinal/v02_model.hdf5")
    f = h5py.File("/home/s4928793/RDProject/datasetFiles/data_sNs.hdf5", 'r')
else: 
    model = keras.models.load_model("/Users/moirashooter/RDProject/modelsFinal/v01_model.hdf5")
    f = h5py.File("/Users/moirashooter/RDProject/datasetFiles/data_sNs.hdf5", 'r')
# get the data
images = f['dataset_images']
labels = f['dataset_labels']
# strore them in np.arrays 

np_images = np.array(images.value)
np_labels = np.array(labels.value)
# do something with data
f.close()
 
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
# print model information 
model.summary()
model_json = model.to_json()
with open("/home/s4928793/Desktop/model.json", "w") as json_file:
    json_file.write(model_json)

#start = time.time()
#with tf.device('/gpu:0'):
#    history = model.evaluate(images_val, label_val, verbose=1)
#end = time.time() 
#print("model took %0.2f seconds to evaluate" % (end-start))
#print("%s: %0.4f" % (model.metrics_names[0], history[0]))
#print("%s: %0.4f" % (model.metrics_names[1], history[1]))
