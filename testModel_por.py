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

model=keras.models.load_model("/home/s4906706/Documents/RaD/data/por_ModelV4.hdf5")

f = h5py.File('/home/s4906706/Documents/RaD/data/tibia-mouse3-jpg.hdf5')

images = f['dataset_images']
np_images = np.array(images.value)
np_images = np_images.reshape(np_images.shape[0], 150, 150, 1).astype('float32')
print "reshapeWorked"
prediction = model.predict_classes(np_images)
print prediction

test_labels=['[0-5]','[5-10]','[10-15]','[15-20]','[20-25]','[25-30]','[30-35]','[35-40]','[40-45]','[45-50]',
'[50-55]','[55-60]','[60-65]','[65-70]','[70-75]','[75-80]','[80-85]','[85-90]','[90-95]','[95-100]']

pltSize=100#len(prediction)

pltSkipArray=[]

for i in range(pltSize):
    if i%5==0:
        pltSkipArray.append(i)
    else:
        pltSkipArray.append(" ")

plt.figure(figsize=(20,10))

plt.grid(False)
plt.xticks(range(pltSize),pltSkipArray, rotation=45)
plt.yticks(range(19), test_labels)
  
plt.plot(range(pltSize),prediction[0:pltSize], '-', linewidth=1, label='quadratic')

plt.savefig('/home/s4906706/Desktop/image60Plot.png')