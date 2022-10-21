import numpy as np
import cv2 as cv
import os

def imresize(path, filename):
    img = cv.imread(path+'/'+filename)
    res = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
    cv.imwrite(path+'_2/'+filename, res)

if __name__ == "__main__":
    for file in ["trainsets/trainH", "trainsets/trainL", "testsets/testH", "testsets/testL"]:
        dirs = os.listdir(file)
        for dir_ in dirs:
           imresize(file, dir_)