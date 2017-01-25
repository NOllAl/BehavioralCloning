import cv2

def preproc_image(x):
    x = x[32:135, :, :] 
    x = cv2.resize(x, (208, 66))
    return x