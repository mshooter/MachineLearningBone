##-----------------------------------------------------------------------------------------------
## author: Moira Shooter
## filename: image_converter.py 
## brief: Converts all the images from .bmp to .jpg
##-----------------------------------------------------------------------------------------------
from PIL import Image 
import os 
from process_img import *


##-----------------------------------------------------------------------------------------------
## my variables
prefixpth = '/Users/moirashooter/RDProject/dataset/'
suffix = '-jpg'
# list of all folders
# normal mice 
# sick mice
# transplanted mice
name_folders = ["tibia-mouse1", "tibia-mouse2", "tibia-mouse3", "tibia-mouse4", "tibia-mouse5", "tibia-mouse6", "iom1", "iom2", "iom3", "iom4", "iom5", "iom6", "ts1", "ts2", "ts3", "ts4", "ts5", "ts6", "ts7", "ts8", "ts9"]
# destination folder
destination_folders_path = []
source_folders_path = []
##-----------------------------------------------------------------------------------------------
## stores all the paths into a array 
##-----------------------------------------------------------------------------------------------
def setUpPaths():
    # set up folder 
    for i in name_folders:
        destination_folders_path.append(prefixpth+i+suffix)
        source_folders_path.append(prefixpth+i)
##-----------------------------------------------------------------------------------------------
## creates folders with name-jpg
##-----------------------------------------------------------------------------------------------
def createJPGFolders():
    #  create jpg folders 
    for k in destination_folders_path:
        try:
            if not os.path.exists(k+'/'):
                os.makedirs(k+'/')
        except OSError:
            print('Error creating directry ' +  k +'/')
##-----------------------------------------------------------------------------------------------
## converts all the images into jpg
##-----------------------------------------------------------------------------------------------
def converter():
    # go through every path 
    for folder in source_folders_path:
        if os.path.exists(folder):
            print(folder)
            destination_folder = folder+suffix
            source_folder = folder
            for filename in os.listdir(source_folder):
                if filename.endswith(".bmp"):
                    if os.path.isfile(source_folder+'/'+filename):
                         img = Image.open(source_folder+'/'+filename).convert('RGB').save(destination_folder+'/'+filename[:-3]+'jpg')
##-----------------------------------------------------------------------------------------------
## calls function to create the paths
##-----------------------------------------------------------------------------------------------
setUpPaths()
##-----------------------------------------------------------------------------------------------
## stores the images based on their porosity
##-----------------------------------------------------------------------------------------------
for folder in destination_folders_path:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            path = folder+'/'+filename
            values = process_imgage(path)
            test_image = values[0]
            saveToFolder(values[1], path, test_image)
