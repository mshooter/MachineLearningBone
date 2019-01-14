import numpy as np 
import h5py 
import cv2 
import os 
# hot_encoded
def hot_encoded(directory):
    if directory == 'sick':
        return 1
    if directory == 'not_sick':
        return 0
# get the data set 
# label is not sick = [1,0] and sick = [0,1]
main_path = "/Users/moirashooter/RDProject/dataset_sick_not_sick"
label = []
images = [] 

# set up the labels and images in an list
def createAssignedDataset():
    for subdir, dirs, files in os.walk(main_path):
        for pfile in files:
            if(pfile != '.DS_Store'):
                filename_path = os.path.join(subdir,pfile)
                dirname_path = os.path.dirname(os.path.realpath(os.path.join(subdir, pfile)))
                dirname = os.path.basename(os.path.normpath(dirname_path))
                label.append(hot_encoded(dirname))
                images.append(filename_path)

createAssignedDataset()
image_shape = (len(images), 1,150,150)
# create hdf5 file 
h_file = h5py.File('/Users/moirashooter/RDProject/datasetFiles/data_sNs.hdf5', 'w')
# write data into file
h_file.create_dataset('dataset_images', image_shape, np.int8)
h_file.create_dataset('dataset_labels', (len(images),), np.int8)
h_file['dataset_labels'][...] = label

for i in range(len(images)):
    addr = images[i]
    img = cv2.imread(addr,0)
    img = cv2.resize(img, (150,150))
    h_file["dataset_images"][i, ...] = img[None]
# close file
h_file.close()
