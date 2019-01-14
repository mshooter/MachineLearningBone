import PIL
from PIL import Image 
import os

for filename in os.listdir("/home/s4928793/RDProject/dataset_sick_not_sick/not_sick"):
    if filename.endswith('.bmp'):
        img = Image.open("/home/s4928793/RDProject/dataset_sick_not_sick/not_sick/" + filename).convert('RGB').save("/home/s4928793/RDProject/dataset_sick_not_sick/not_sick/" + filename[:-3] + 'jpg')
        os.remove("/home/s4928793/RDProject/dataset_sick_not_sick/not_sick/" + filename)
