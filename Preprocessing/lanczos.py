import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = "data" 

for img in os.listdir( train_dir + "/original"):
    img_array = cv2.imread(train_dir + "/original/" + img)
    
    img_array = cv2.resize(img_array, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    lr_img_array = cv2.resize(img_array, (32, 32), interpolation=cv2.INTER_LANCZOS4)
    
    cv2.imwrite(train_dir+ "/lanczos/"+ img, lr_img_array)
    cv2.imwrite(train_dir+ "/lanczos/"+ img, lr_img_array)



  










   

