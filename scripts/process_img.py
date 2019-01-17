##----------------------------------------------------------------------------
## author:Moira Shooter
## filename: process_img.py
## build: is one of the scripts we used to process the image to calculate the porosity factor 
##----------------------------------------------------------------------------
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import os 

#src = cv2.imread("/Users/moirashooter/RDProject/dataset/tibia-mouse2-jpg/26Oim__rec0002.jpg",0) 
#src = cv2.imread("/Users/moirashooter/RDProject/dataset/ts5-jpg/118Oim__rec0200.jpg",0) 
##----------------------------------------------------------------------------
# function for preparing the data set
# param : path to the picture 
##----------------------------------------------------------------------------
def createImage(pathToImage):
    # read image 
    # image is a matrix 
    src = cv2.imread(pathToImage, 0)
    cv2.imshow("src", src)
    # blur to get rid of noise 
    blur = cv2.blur(src, (8, 8))
    cv2.imshow("blur", blur)
    # thresholding
    ret, im_thresh1 = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold", im_thresh1)
    blur2= cv2.blur(im_thresh1, (8,8))
    cv2.imshow("blur2", blur)
    ret, im_thresh2 = cv2.threshold(blur2, 50, 255, cv2.THRESH_BINARY)
    mask_alpha = im_thresh2
    cont = im_thresh1
    # opening contour
    im2,  contours1, hierarchy = cv2.findContours(cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = [] 
    for i in range(len(contours1)):
        hull.append(cv2.convexHull(contours1[i], False))
    drawing = np.zeros((cont.shape[0], cont.shape[1], 3), np.uint8)
    for i in range(len(contours1)):
        color_contour = (0,255,0)
        color = (255, 0, 0)
        #cv2.drawContours(drawing, contours1, i, color_contour, 1, 8, hierarchy)
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    cv2.imshow("t", drawing)
    # creating a mask_alpha 
    img, contour, hier = cv2.findContours(mask_alpha, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask_alpha,[cnt],0,255, -1)
    cv2.imshow("mask_alpha",mask_alpha)
    # outline of the bone 
    #im_outline = cv2.Canny(mask_alpha, 0, 255, 3)
    # merge the two images together 
    # img_merge = cv2.addWeighted(src, 0.8, im_outline, 0.5, 0)
#    ret, img_thresh_final = cv2.threshold(src, 90, 255, cv2.THRESH_BINARY)
#    img_merge2 = cv2.addWeighted(img_thresh_final, 0.8, cv2.bitwise_not(mask_alpha), 0.2, 0)
#    return img_merge2
 # calculate the porosity of an image 
##----------------------------------------------------------------------------
## function that calculates the porosity
##----------------------------------------------------------------------------
def calculatePorosity(img):
    # amount of black and white pixels 
    n_white_pix = np.sum(img == 204)
    n_black_pix = np.sum(img == 0) 
    # error on the thingie - fix it. 
    n_total_black_pix = n_black_pix - (n_black_pix*0.20)
    print('amount of white pixels: ', n_white_pix)
    print('amount of black pixels: ',n_total_black_pix)
    if(n_total_black_pix != 0):
        porosity = n_white_pix / n_total_black_pix
    else:
        porosity = 0 
        return porosity
    print('porosity factor: ', porosity) 
    return porosity * 100
    
##----------------------------------------------------------------------------
## function that gets the filename path
##----------------------------------------------------------------------------
def getFileName(path):
    base = os.path.basename(path)
    return base
original_path = '/Users/moirashooter/RDProject/dataset_final/'
# if porosity is a certain range - store it in directory 
def saveToFolder(porosity, pathToImage, img):
    dest_path =  original_path
    lower_bound = int(porosity / 5) * 5
    upper_bound = lower_bound + 5
    path = dest_path+'[{}-{}]'.format(lower_bound, upper_bound)+'/'+getFileName(pathToImage)
    cv2.imwrite(path, img)

##----------------------------------------------------------------------------
## path variables, this was to test
##----------------------------------------------------------------------------
paths = "/Users/moirashooter/RDProject/dataset/tibia-mouse2-jpg/26Oim__rec0002.jpg"
pathss = "/Users/moirashooter/RDProject/dataset/tibia-mouse6-jpg/142Oim__rec0120.jpg"
pathst = "/Users/moirashooter/RDProject/dataset/iom3-jpg/22Oim__rec0010.jpg"
patt ="/Users/moirashooter/RDProject/dataset/tibia-mouse5-jpg/141Oim__rec0991.jpg"
##----------------------------------------------------------------------------
## This function process the image, to find the area we are interested in, 
## in creates the alpha mask - another function 
##----------------------------------------------------------------------------
def process_imgage(path):
    print(path)
    # create the alpha mask
    if os.path.exists(paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        ret, img_thresh_final = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
#         ret, th = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
#         output = np.hstack([img, cv2.bitwise_not(th)])
             
        blur = cv2.blur(img, (6,6))
        ret, threshold = cv2.threshold(blur , 50, 255, cv2.THRESH_BINARY)

        blur1 = cv2.blur(threshold, (25,25))
        ret, threshold1 = cv2.threshold(blur1 , 50, 255, cv2.THRESH_BINARY)
        blur2 = cv2.blur(threshold1,(3,3))
        ret, threshold2 = cv2.threshold(blur2, 50, 255, cv2.THRESH_BINARY)
         
        img2, contour, hier = cv2.findContours(threshold2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
         

        for cnt in contour:
            cv2.drawContours(threshold2,[cnt],0,255, -10)
        finalImg = cv2.addWeighted(cv2.bitwise_not(threshold2), 0.2, img_thresh_final,0.8, 0) 
        porosity = calculatePorosity(finalImg)
        img = cv2.resize(img, (150, 150))
        return [img, porosity]

