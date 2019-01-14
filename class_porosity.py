import os 
import sys 
sys.path.append("/home/s4928793/RDProject/dataset_sick_not_sick/scripts")
import porous
import cv2

filepath ="/home/s4928793/RDProject/dataset_sick_not_sick/not_sick" 
storepath ="/home/s4928793/RDProject/dataset_pf/" 
 
for filename in os.listdir(filepath):
    filepath_name = os.path.join(filepath + '/' + filename)
    porosity = porous.proccesImage(filepath_name)[1]
    final_img = cv2.imread(filepath_name, 0)
    # check directory porosity 
    for directory in os.walk(storepath):
        porosityDir = os.path.basename(directory[0])
        if str(porosity) == str(porosityDir):
            print(directory[0] + '/' + filename)
            cv2.imwrite(directory[0] + '/' + filename, final_img)     
