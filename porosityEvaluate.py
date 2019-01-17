import os
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

model=keras.models.load_model("/home/s4906706/Documents/RaD/data/por_ModelV4.hdf5")
f = h5py.File("/home/s4906706/Documents/RaD/data/processed_data/por_Data.hdf5")

images = f["dataset_images"]
labels = f["dataset_labels"]

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

history=model.evaluate(images_val, label_val, verbose=1)

print("%s: %0.4f" % (model.metrics_names[1], history[1]))

# plt.imshow(images_train[1])
# plt.savefig("/home/s4906706/Documents/RaD/processed_data/")
# print class_names[labels[0]]
