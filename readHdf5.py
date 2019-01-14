import h5py 
import numpy as np 
import cv2
f = h5py.File("/Users/moirashooter/RDProject/datasetFiles/data_sNs.hdf5", 'r')

#dataset = f['dataset_2']
# get the data
images = f['dataset_images']
labels = f['dataset_labels']
np_images = np.array(images.value, np.int8)
np_label = np.array(labels.value, np.int8)
# strore them in np.arrays 
cv2.imwrite("/Users/moirashooter/Desktop/test.jpg", np_images[0][0])
# do something with data
f.close()

