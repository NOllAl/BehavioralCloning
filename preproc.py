import cv2
import numpy as np

def preproc_image(x):
    #x = cv2.cvtColor(x, cv2.COLOR_BGR2YUV)
    x = x[32:135, :, :] 
    #x = x[32:135, 40:280, :] 
    x = cv2.resize(x, (208, 66))
    x = x / 255 - 0.5
    x = np.reshape(x, (1, 66, 208, 3))
    return x