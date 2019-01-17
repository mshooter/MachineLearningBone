###-----------------------------------------------------------------------------------------------------------------------------------
## author: Moira Shooter
## nameOfFile: visualise_images.py
###-----------------------------------------------------------------------------------------------------------------------------------
import time
import os
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np
#from sklearn.cross_validation import train_test_split
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

#----------------------------------------------------------------------------------------------------------------------------
# my variables
#----------------------------------------------------------------------------------------------------------------------------
linux = True
class_names=['healthy', 'sick']
#----------------------------------------------------------------------------------------------------------------------------
# get the dataset it is stored in a hdf5
#----------------------------------------------------------------------------------------------------------------------------
import h5py 
if linux:
    f = h5py.File("/home/s4928793/RDProject/datasetFiles/data_sNs.hdf5", 'r')
else: 
    f = h5py.File("/Users/moirashooter/RDProject/datasetFiles/data_sNs.hdf5", 'r')
# get the data
with tf.device('/gpu:0'):
    images = f['dataset_images']
    labels = f['dataset_labels']
# store data in np
np_images = np.array(images.value)
np_labels = np.array(labels.value)
f.close()
#----------------------------------------------------------------------------------------------------------------------------
# split into training and test set (eg 80/20)
images_train, images_val, label_train, label_val = train_test_split(np_images, np_labels, test_size = 0.1, shuffle = True)
images_train = images_train/255.0
# hot encoded labels
label_train = keras.utils.to_categorical(label_train)
label_val = keras.utils.to_categorical(label_val)
#----------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,10))
for i in range(30):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_train[i][0], cmap=plt.cm.binary)
    plt.xlabel(class_names[int(label_train[i][0])])
plt.savefig('/home/s4928793/Desktop/train_fig.jpg')
