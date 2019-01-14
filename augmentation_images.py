import cv2
import random
import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image, ImageEnhance

# paths for folders training 
not_sick_path = "/Users/moirashooter/RDProject/dataset_sick_not_sick/not_sick"
sick_path = "/Users/moirashooter/RDProject/dataset_sick_not_sick/sick"

# debug function
def test():
    trans_range = 5
    img = cv2.imread("/Users/moirashooter/RDProject/dataset_sick_not_sick/not_sick/142Oim__rec0973.jpg",0)
    rows, cols = img.shape
    # augment 
    tr_x = trans_range * np.random.uniform() - trans_range/2
    tr_y = trans_range * np.random.uniform() - trans_range/2
    trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # write file
    print(img.shape)
test()
# final function 
def imageFlip(path):
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            # augment
            img = cv2.imread(path+"/"+file)
            # y - axis
            flipped_imgh = cv2.flip(img, 0)
            flipped_imgv = cv2.flip(img, 1)
            flipped_imgb = cv2.flip(img, -1)
            cv2.imwrite(path + "/flh_" + file, flipped_imgh)
            cv2.imwrite(path + "/flv_" + file, flipped_imgv)
            cv2.imwrite(path + "/flb_" + file, flipped_imgb)

def imageTranslate(path):
    # trans_range: range f values to apply translatios over 
    trans_range = 300 
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img = cv2.imread(path+"/"+file)
            # augment 
            tr_x = trans_range * np.random.uniform() - trans_range/2
            tr_y = trans_range * np.random.uniform() - trans_range/2
            trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
            # write file
            img = cv2.warpAffine(img, trans_M, (960,960))
            cv2.imwrite(path + "/t_" +file, img)
                        
def random_brightness(path):
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img = Image.open(path+"/"+file)
            randomNumb = round(random.uniform(0.3,2), 1)
            contrast = ImageEnhance.Contrast(img)
            contrast = contrast.enhance(randomNumb)
            contrast.save(path + "/con_" + file)

# main function
