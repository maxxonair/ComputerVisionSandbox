import numpy as np
import glob
import os
from tqdm import tqdm
import cv2 as cv

def downsampleImage(image, reductionFactor):
	for i in range(0,reductionFactor):
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape
		image = cv.pyrDown(image, dstsize= (col//2, row // 2))
	return image

# Function to remove all files in given folder:
def cleanFolder(folderPath):
    files = glob.glob(folderPath+'*')
    for f in files:
        os.remove(f)