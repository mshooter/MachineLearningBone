###-------------------------------------------------------------------------------------------------------
## author: Moira Shooter
## fileName: application.py 
## brief: This script was to merge the two neural networks togeter to 3D reconstruct the model 
###-------------------------------------------------------------------------------------------------------
import h5py
import tensorflow as tf
from tensorflow import keras 
import numpy as np

#----------------------------------------------------------------------------------------------------------------------------
# my variables
# health factor is sick bone
health_factor = 1
#----------------------------------------------------------------------------------------------------------------------------
# load two models
#----------------------------------------------------------------------------------------------------------------------------
model_health = keras.models.load_model("/home/s4928793/Desktop/RDFinal/models/v03_model_health.hdf5")
model_porosity = keras.models.load_model("/home/s4928793/Desktop/RDFinal/models/por_ModelV4.hdf5")
#----------------------------------------------------------------------------------------------------------------------------
# get a dataset
#----------------------------------------------------------------------------------------------------------------------------
f = h5py.File('/home/s4928793/Desktop/oneBone.hdf5')
images = f['dataset_images']
np_images = np.array(images.value)
f.close()
#----------------------------------------------------------------------------------------------------------------------------
# reshape 
print(np_images.shape[0])
#----------------------------------------------------------------------------------------------------------------------------
# sequential of the two models predicting what it is  
#----------------------------------------------------------------------------------------------------------------------------
# prediciting if it is sick 
# need to run the porosity function to have the labels 
# if there is more sick than not sick then the bone sick, visa versa
#----------------------------------------------------------------------------------------------------------------------------
# print the result -> 
# predicting the porosity factor 
# print the result -> 

