import cv2

def preproc_image(x):
    x = cv2.resize(x, (160, 80))
    return x