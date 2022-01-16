# -------------------------------------------------------------------------------------------
#               >> Camera calibration 
# -------------------------------------------------------------------------------------------
# 
# From:
# https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
# 
# Setup:
# 
# So to find pattern in chess board, we can use the function, cv.findChessboardCorners(). 
# We also need to pass what kind of pattern we are looking for, like 8x8 grid, 5x5 grid etc. 
# In this example, we use 7x6 grid. (Normally a chess board has 8x8 squares and 7x7 internal 
# corners). It returns the corner points and retval which will be True if pattern is obtained. 
# These corners will be placed in an order (from left-to-right, top-to-bottom)
# 
# 
# -------------------------------------------------------------------------------------------
import numpy as np
import os
import cv2 as cv
import glob
from tqdm import tqdm
from supportFunctions import cleanFolder
# -------------------------------------------------------------------------------------------
#               >> Settings
# -------------------------------------------------------------------------------------------
# Save images with identified corners drawn 
savePlot        = True
# Enable input image scaling (scale factor below)
enableScaling   = True
# Enable compute reprojection error:
computeReprojectionError = True

#Define size of chessboard target.
boardSize = (8,5)

# Process images at scale: 
scale_percent = 50

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# -------------------------------------------------------------------------------------------
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((boardSize[1]*boardSize[0],3), np.float32)
objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Flag to inicate that corners have been found 
isSuccess       = False
successCounter  = 0 
imageIndex      = 0 

# Set file paths: 
parameterFilePath   = "./03_camera_parameters/"
processedImagePath  = "./02_processed_calibration_images/"
inputFilePath       = "./01_calibration_images/"

# List calibration images:
images = glob.glob(inputFilePath+"*")

# Empty folder for processed images 
cleanFolder(processedImagePath)
# Empty folder for calibration parameters:
cleanFolder(parameterFilePath)

for fname in tqdm(images):
    # Load image 
    img         = cv.imread(fname)
    # Convert image to greyscale 
    gray        = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Scale image 
    if enableScaling:
        width  = int(gray.shape[1] * scale_percent /100 )
        height = int(gray.shape[0] * scale_percent /100 )
        dsize = (width, height)
        outputImage = cv.resize(gray, dsize)
        outputColor = cv.resize(img, dsize)
    else:
        outputImage = gray
        outputColor = img

    # Print status
    fileName    = fname.split("/")
    fileName    = fileName[-1]
    print('Load: ' + fileName)
    print("Image loaded, Analizying...")

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(outputImage, boardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        corners2    = cv.cornerSubPix(outputImage,corners, (11,11), (-1,-1), criteria)
        isSuccess   =  True
        successCounter = successCounter + 1
        # Save scaled images with drawn corners
        if savePlot:
            cv.drawChessboardCorners(outputColor, boardSize, corners2, ret)
            savePath = processedImagePath + fileName
            print("Result images saved to: "+savePath)
            cv.imwrite(savePath, outputColor)
    else:
        print("")
        print(" >> ("+str(imageIndex)+")  No chessboard found.")
        print("")
    imageIndex = imageIndex + 1

print(str(successCounter) + " Images processed. "+str(successCounter/imageIndex * 100)+" percent. ")           


if isSuccess:
    # Create camera calibration parameters 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, outputImage.shape[::-1], None, None)
    #Save parameters into numpy file
    np.save(parameterFilePath+"ret", ret)
    np.save(parameterFilePath+"K", mtx)
    np.save(parameterFilePath+"dist", dist)
    np.save(parameterFilePath+"rvecs", rvecs)
    np.save(parameterFilePath+"tvecs", tvecs)

    mean_error = 0
    if computeReprojectionError:
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "Total reprojection error: {}".format(mean_error/len(objpoints)) )

    print("")
    print(' >>> Camera parameters saved succesfully!)')
    print("")

else:
    print("")
    print(' >>> Calibration failed (No chessboard found)')
    print("")