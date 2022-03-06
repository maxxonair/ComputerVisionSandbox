# -------------------------------------------------------------------------------------------
#               >> Camera image correction
# -------------------------------------------------------------------------------------------
# 
# From:
# https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
# 
# 
# 
# -------------------------------------------------------------------------------------------
import numpy as np
import cv2 as cv
import glob
import os
from tqdm import tqdm
# -------------------------------------------------------------------------------------------

useRemapping = True
# Set file paths: 
parameterFilePath   = "./03_camera_parameters/"
resultFilePath      = "./05_corrected_images/"
inputFilePath       = "./06_scaled_images/"

ret     = np.load(parameterFilePath+"ret.npy")
mtx     = np.load(parameterFilePath+"K.npy")
dist    = np.load(parameterFilePath+"dist.npy")
rvecs   = np.load(parameterFilePath+"rvecs.npy")
tvecs   = np.load(parameterFilePath+"tvecs.npy")

# Function to remove all files in given folder:
def cleanFolder(folderPath):
    files = glob.glob(folderPath+'*')
    for f in files:
        os.remove(f)

# Empty folder for processed images 
cleanFolder(resultFilePath)

# List calibration images:
images = glob.glob(inputFilePath+"*")

for fname in tqdm(images):
    # Load image 
    img         = cv.imread(fname)
    h,  w = img.shape[:2]

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    if useRemapping:
        # undistort
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    else:
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        
    # crop the image
    x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]

    # Print status
    fileName    = fname.split("/")
    fileName    = fileName[-1]
    fileName    = fileName.split(".")
    fileName    = fileName[0]
    #print("")
    #print('Save Result file : ' + fileName)
    cv.imwrite(resultFilePath+fileName+".png", dst)

