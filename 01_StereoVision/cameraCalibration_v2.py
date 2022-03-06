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
# import numpy as np
# import os
# import cv2 as cv


from supportFunctions import cleanFolder
from PyLog import PyLog
from Calibrator import Calibrator
# -----------------------------------------------------------------------------
#               >> Settings
# -----------------------------------------------------------------------------
# Set file paths: 
parameterFilePath   = "./03_camera_parameters/"
processedImagePath  = "./02_processed_calibration_images/"
inputFilePath       = "./01_calibration_images/"
scaledImgPath       = "./06_scaled_images/"
# Save images with identified corners drawn 
savePlot                 = True
# Save scaled images 
enableSaveScaledImgs     = True
# Enable input image scaling (scale factor below)
enableScaling            = True
# Enable compute reprojection error:
computeReprojectionError = True

#Define size of chessboard target.
# Number of inner corners per a chessboard row and column 
#( patternSize = cvSize (points_per_row, points_per_colum) = cvSize(columns,rows) ).
boardSize = (9,6)
# -----------------------------------------------------------------------------
#   >> Setup Logger
# -----------------------------------------------------------------------------
# Flag: Enable console prints 
flagIsConsolePrint = True ;
# Flag: Create and save logging text file
flagIsSaveLogFile  = False ;
log = PyLog("CalibrationLog", flagIsConsolePrint, flagIsSaveLogFile);
# -----------------------------------------------------------------------------
# Empty folder for processed images 
cleanFolder(processedImagePath)
# Empty folder for calibration parameters:
cleanFolder(parameterFilePath)
#------------------------------------------------------------------------------
# >> Calibrate
#------------------------------------------------------------------------------
# Initialize calibration
calibrator = Calibrator(inputFilePath, boardSize, log)

# Run single camera calibration
calibrator.calibrate()

# Save Calibration Results:
calibrator.saveResults()
#------------------------------------------------------------------------------ 
log.close()
