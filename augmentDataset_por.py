#import /home/s4906706/Documents/RaD/MachineLearningBone/augmentation_images.py
import cv2
import random
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import h5py
from PIL import Image, ImageEnhance

folderpath = '/home/s4906706/Documents/RaD/data/dataset_final'

folder=[]
imgnum=[]
images=[]
imagelabels=[]
tempimages=[]

test=True

labels=['[0-5]','[5-10]','[10-15]','[15-20]','[20-25]','[25-30]','[30-35]','[35-40]','[40-45]',
'[45-50]','[50-55]','[55-60]','[60-65]','[65-70]','[70-75]','[75-80]','[80-85]','[85-90]',
'[90-95]','[95-100]']

#-----------------------------------------Img Flip Augmentation----------------------

def imageFlip(img, axis):

    if axis == 0:
        img = cv2.flip(img, 0)
    if axis == 1:
        img = cv2.flip(img, 1)
    if axis == -1:
        img = cv2.flip(img, -1)
    return img

#-----------------------------------------Img Translate Augmentation-----------------

def imageTranslate(img):
    trans_range = 50
    tr_x = trans_range * np.random.uniform() - trans_range/2
    tr_y = trans_range * np.random.uniform() - trans_range/2
    trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]]) 
    img = cv2.warpAffine(img, trans_M, (150,150))
    return img

def imageRotate(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat= cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

#-----------------------------------------Test image augmentation functions----------

if test == True:
    testfile = '/home/s4906706/Documents/RaD/data/dataset_final/[90-95]/137Oim__rec0920.jpg'
    testimg = cv2.imread(testfile,0)
    for i in range(15):
        augmentedimg= imageFlip(testimg, -1)
        augmentedimg = imageTranslate(augmentedimg)
        rotNum=random.randrange(1, 23)*15
        augmentedimg = imageRotate(augmentedimg, rotNum)
        cv2.imwrite("/home/s4906706/Desktop/testimage.jpg", testimg)
        cv2.imwrite("/home/s4906706/Desktop/augmentedimage"+ str(i) + ".jpg", augmentedimg)

#-----------------------------------------Main Function------------------------------

if test!=True:

    for j in os.listdir(folderpath):

        rangefolderpath=folderpath + '/' + j            #get range folder
        p=0                                             #p refers to total amount of img in folder
        labelnum=0
        for t in labels:
            #print t
            #print j
            if j == t:
                labelnum=labels.index(j)
    
        for i in os.listdir(rangefolderpath):           
            imgpath=rangefolderpath + '/' + i
            if imgpath.endswith(".jpg"):
                tempimages.append(cv2.imread(imgpath,0))
                p+=1
    
        if p<1000:          #if amount of images in this range is smaller than 1000, augment the dataset for that folder to 1000 images
            #print "smaller"
            while len(tempimages)!=1000:
                a = random.randrange(0, p)
                randimg =rangefolderpath + '/' + os.listdir(rangefolderpath)[a]
                img = cv2.imread(randimg,0)
                flipNum=random.randrange(-1,1)
                rotNum=random.randrange(1, 23)*15
                img = imageFlip(img, flipNum)
                img = imageTranslate(img)
                img = imageRotate(img, rotNum)
                tempimages.append(img)
        else:               #if amount of images in this range is bigger than 1000, reduce it at random
            #print "bigger"
            while len(tempimages)!=1000:
                a = random.randrange(0, len(tempimages))
                del tempimages[a]
        #cv2.imwrite("/home/s4906706/Desktop/testimage.jpg", tempimages[500])
        #print len(tempimages)
    
        for t in tempimages:
            images.append(t)
            imagelabels.append(labelnum)
            #print labelnum
    #----------------------------------------delete temporary images-----------------------------    
        del tempimages[:]
    print len(images)
    print len(imagelabels)
    image_shape = (len(images),1,150,150)
    
    # create hdf5 file
    h_file = h5py.File('/home/s4906706/Documents/RaD/data/processed_data/por_Data.hdf5', 'w')
    h_file.create_dataset('dataset_images', image_shape, np.int8)
    h_file.create_dataset('dataset_labels', (len(images),), np.int8)
    h_file['dataset_labels'][...] = imagelabels
    for i in range(len(images)):
        h_file['dataset_images'][i, ...] =images[i] 
    
    h_file.close()
 