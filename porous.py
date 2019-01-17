###-----------------------------------------------------------------------------------------------------------------------------------
## author: Moira Shooter
## nameOfFile: porous.py
## brief: The script was used to calculate the porosity and preprocess the image to find the alpha mask, so we only focus on the 
## area of interest 
###-----------------------------------------------------------------------------------------------------------------------------------
import numpy as np 
import cv2 

path_test = "/home/s4928793/RDProject/dataset_sick_not_sick/sick/1Oim__rec0200.jpg"
###-----------------------------------------------------------------------------------------------------------------------------------
# calculate the porosity method reversed
###-----------------------------------------------------------------------------------------------------------------------------------
def checkValue(img):
    sum_white = 0
    sum_black = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = img[i][j]
            if pixel == 204: 
                sum_white += 1
            if pixel == 0: 
                sum_black += 1
    porosity = int(float(100 * sum_black / (sum_white + sum_black)))
    return porosity
    #print("amount of white pixels: {}".format(sum_white))
    #print("amount of black pixels: {}".format(sum_black)) 
    print("porosity: {} ".format(porosity))

###-----------------------------------------------------------------------------------------------------------------------------------
# process the image to get the porosity factor
###-----------------------------------------------------------------------------------------------------------------------------------
def proccesImage(path):
    img = cv2.imread(path, 0)
    # get rid of noise
    blur1 = cv2.blur(img, (5,5))
    ret, threshold = cv2.threshold(blur1, 50, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(img, (11,11), 0)
    ret, dst = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    # find contours
    im2, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    color = (255,0,0)
    cv2.drawContours(dst, contours, -1, color, -1)
    # fill holes by morphing
    kernel = np.ones((130,130), np.uint8)
    closing = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    # add the two images to get a gray bg
    out = cv2.addWeighted(cv2.bitwise_not(closing),0.2,threshold,0.8,0)
    porosity = checkValue(out)
    return [out, porosity] 
