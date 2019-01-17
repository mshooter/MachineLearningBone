###-----------------------------------------------------------------------------------------------------------------------------------
## author: Moira Shooter
## nameOfFile: convertToOpenVDB.py
## brief: The use of this script was to convert the 2D slices into a 3D model, the future work would involve including the neural 
## network models to reconstruct the bone procedurally
###-----------------------------------------------------------------------------------------------------------------------------------
import pyopenvdb as vdb
import cv2 as cv
import numpy as np
import os

#----------------------------------------------------------------------------------------------------------------------------
# load the two NN models
#----------------------------------------------------------------------------------------------------------------------------
model_health = keras.models.load_model("/home/s4928793/Desktop/RDFinal/models/v03_model_health.hdf5")
model_porosity = keras.models.load_model("/home/s4928793/Desktop/RDFinal/models/por_ModelV4.hdf5")
#--------------------------------------------------------------------------------------------------------------------------------------
# my variables
path='/home/s4928793/RDProject/dataset_porosity/not_sick/26Oim__rec0059.jpg' 
i_path='/home/s4928793/RDProject/dataset_porosity' 
images = []
#--------------------------------------------------------------------------------------------------------------------------------------
# iterate over every file, and store in array
#--------------------------------------------------------------------------------------------------------------------------------------
for filename in os.listdir(i_path):
    if filename.endswith(".jpg"):
        images.append(i_path+'/'+filename)
#--------------------------------------------------------------------------------------------------------------------------------------
# how to read one image. need to get rid of the noise 
#--------------------------------------------------------------------------------------------------------------------------------------
def readOneImage(_path):
    img = cv.imread(_path,0)
    blur = cv.blur(img, (5,5))
    blur = cv.resize(blur, (150,150))
    ret, thresh1 = cv.threshold(blur,40,255,cv.THRESH_BINARY);
    return thresh1    
#--------------------------------------------------------------------------------------------------------------------------------------
# create a 3D gird with value 0 
# iterate over image check if the value is white, if it is fill the voxel with value 1
#--------------------------------------------------------------------------------------------------------------------------------------
size=150
cube = vdb.FloatGrid()
cube.fill((0,0,0), (size,size,size), 0)
acc = cube.getAccessor()
i = 0
for image in images: 
    print(image)
    im=readOneImage(image) 
    for x in range(0,size):
        print(x)
        for y in range(0,size):
            # this is the depth 
            # adds new image
            if im[x,y] == 255:
                 acc.setValueOnly((i,y,x), 1.0)
    i=i+1
# Write both grids to a VDB file.
vdb.write('/home/s4928793/Desktop/healthy.vdb', grids=[cube])

