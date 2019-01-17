###-----------------------------------------------------------------------------------------------------------------------------------
## author: Moira Shooter
## nameOfFile: createHdf5.py
## brief: This was to create the dataset with images including the label, easier way to use the dataset in other files by loading 
## the .hdf5 file
###-----------------------------------------------------------------------------------------------------------------------------------
import numpy as np 
import h5py 
import cv2 
import os 
#-------------------------------------------------------------------------------------------------------
# variables
#-------------------------------------------------------------------------------------------------------
main_path = "/Users/moirashooter/RDProject/dataset_sick_not_sick"
main_path_1 = "/home/s4928793/RDProject/oneSickBone"
label = []
images = [] 
#-------------------------------------------------------------------------------------------------------
# hot_encoded method
#-------------------------------------------------------------------------------------------------------
def hot_encoded(directory):
    if directory == 'sick':
        return 1
    if directory == 'not_sick':
        return 0
#-------------------------------------------------------------------------------------------------------
# set up the labels and images in an list
#-------------------------------------------------------------------------------------------------------
def createAssignedDataset():
    for subdir, dirs, files in os.walk(main_path):
        for pfile in files:
            if(pfile != '.DS_Store'):
                filename_path = os.path.join(subdir,pfile)
                dirname_path = os.path.dirname(os.path.realpath(os.path.join(subdir, pfile)))
                dirname = os.path.basename(os.path.normpath(dirname_path))
                label.append(hot_encoded(dirname))
                images.append(filename_path)
#-------------------------------------------------------------------------------------------------------
# set up the labels and images in an list, traverse all files in folder
#-------------------------------------------------------------------------------------------------------
def createHdf5():
    for filename in os.listdir(main_path_1):
        if filename.endswith(".jpg"):
            filename_path =  main_path_1 + '/' +filename 
            images.append(filename_path)
#-------------------------------------------------------------------------------------------------------
# run the function 
#-------------------------------------------------------------------------------------------------------
#createAssignedDataset()
createHdf5()
#-------------------------------------------------------------------------------------------------------
### The following section is from :-
### Machine Learning Guru. Saving and loading a large number of images (data) into a single HDF5 file[online]. [Accesses 2018]
### Available from: "http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html".
### Edited: to resize the images, and set them black and white (grayscale)
#-------------------------------------------------------------------------------------------------------
image_shape = (len(images), 1,150,150)
# create hdf5 file 
h_file = h5py.File('/home/s4928793/Desktop/oneBone.hdf5', 'w')
# write data into file
h_file.create_dataset('dataset_images', image_shape, np.int8)
#h_file.create_dataset('dataset_labels', (len(images),), np.int8)
#h_file['dataset_labels'][...] = label

for i in range(len(images)):
    addr = images[i]
    img = cv2.imread(addr,0)
    img = cv2.resize(img, (150,150))
    h_file["dataset_images"][i, ...] = img[None]
# close file
h_file.close()
### end of citation
