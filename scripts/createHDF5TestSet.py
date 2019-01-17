import os
import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

testSetPath = '/home/s4906706/Documents/RaD/data/tibia-mouse3-jpg'

model=keras.models.load_model("/home/s4906706/Documents/RaD/data/por_ModelV4.hdf5")

images=[]

for i in os.listdir(testSetPath):
    if i.endswith("DS_Store") or i.startswith("."):
        print "wrongun"
    else:
        img = cv2.imread(testSetPath + "/" + i, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150,150))
        images.append(img)

image_shape = (len(images), 1, 150,150)

h_file=h5py.File('/home/s4906706/Documents/RaD/data/tibia-mouse3-jpg.hdf5', 'w')

h_file.create_dataset('dataset_images', image_shape, np.int8)
h_file.create_dataset('dataset_labels', (len(images),), np.int8)

for i in range(len(images)):
    h_file['dataset_images'][i, ...] =images[i]


h_file.close

print image_shape