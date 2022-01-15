import numpy as np
import cv2 as cv
import glob
from tqdm import tqdm

setDrawPlot     = False
savePlot        = True

#Define size of chessboard target.
chessboard_size = (8,6)

# Process images at scale: 
scale_percent = 35

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Flag to inicate that corners have been found 
isSuccess = False

# Set file paths: 
parameterFilePath   = "./03_camera_parameters/"
processedImagePath  = "./02_processed_calibration_images/"
inputFilePath       = "./01_calibration_images/"

images = glob.glob(inputFilePath+"*")

imageIndex = 0 

for fname in tqdm(images):
    # Load image 
    img         = cv.imread(fname)
    # Convert image to greyscale 
    gray        = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Scale image 
    width  = int(gray.shape[1] * scale_percent /100 )
    height = int(gray.shape[0] * scale_percent /100 )
    dsize = (width, height)
    outputImage = cv.resize(gray, dsize)
    outputColor = cv.resize(img, dsize)

    # Print status
    fileName    = fname.split("/")
    fileName    = fileName[-1]
    print('Load: ' + fileName)
    print("Image loaded, Analizying...")

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(outputImage, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2    = cv.cornerSubPix(outputImage,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        isSuccess   =  True
        # Draw and display the corners
        if savePlot:
            cv.drawChessboardCorners(outputColor, chessboard_size, corners2, ret)
            savePath = processedImagePath + fileName
            print(savePath)
            cv.imwrite(savePath, outputColor)
        if setDrawPlot:
            cv.drawChessboardCorners(outputColor, chessboard_size, corners2, ret)
            cv.imshow('img', outputColor)
            cv.waitKey(500)
    else:
        print(" >> ("+str(imageIndex)+")  No chessboard found.")
    imageIndex = imageIndex + 1

            
#cv.destroyAllWindows()   

if isSuccess:
    # Create camera calibration parameters 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, outputImage.shape[::-1], None, None)
    #Save parameters into numpy file
    np.save(parameterFilePath+"ret", ret)
    np.save(parameterFilePath+"K", mtx)
    np.save(parameterFilePath+"dist", dist)
    np.save(parameterFilePath+"rvecs", rvecs)
    np.save(parameterFilePath+"tvecs", tvecs)
    print("")
    print(' >>> Camera parameters saved succesfully!)')
    print("")

else:
    print("")
    print(' >>> Calibration failed (No chessboard found)')
    print("")