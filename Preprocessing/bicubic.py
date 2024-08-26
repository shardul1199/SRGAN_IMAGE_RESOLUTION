import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = "data" 

   
for img in os.listdir( train_dir + "/original"):
    img_array = cv2.imread(train_dir + "/original/" + img)
    
  
    lr_img_array = cv2.resize(img_array,(32,32), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(train_dir+ "/bicubic/"+ img, lr_img_array)
   

