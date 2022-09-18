#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#               >> Camera calibration 
# -----------------------------------------------------------------------------
# 
# From:
# https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
# 
# Setup:
# 
# So to find pattern in chess board, we can use the function, 
# cv.findChessboardCorners(). We also need to pass what kind of pattern we are 
# looking for, like 8x8 grid, 5x5 grid etc. In this example, we use 7x6 grid. 
# (Normally a chess board has 8x8 squares and 7x7 internal corners). It 
# returns the corner points and retval which will be True if pattern is 
# obtained. These corners will be placed in an order (from left-to-right, 
# top-to-bottom)
# 
# 
# -----------------------------------------------------------------------------
import os
import datetime
import math

from util.supportFunctions import cleanFolder
from util.PyLog import PyLog
from util.Calibrator import Calibrator

def main():
    # -----------------------------------------------------------------------------
    #               >> Folder Setup
    # -----------------------------------------------------------------------------
    camera_ID    = "01"
    raw_img_path = "./_camera_01_calibration/2022_44_28_Aug_08_1661683472_frame_capture/"

    # >> Setup folder structure 
    # Setup folder to save images to 
    dateTimeStr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    folder_name = dateTimeStr+"_camera_"+str(camera_ID)+"_calibration"
    # start time counter 
    path = "./" + folder_name
    os.mkdir(path)

    # Set file paths: 
    parameterFilePath   = path + "/03_camera_parameters/"
    processedImagePath  = path + "/02_processed_calibration_images/"
    inputFilePath       = raw_img_path # path + "/01_calibration_images/"
    scaledImgPath       = path + "/06_scaled_images/"
    sRecifiedImgPath    = path + '/05_corrected_images/'

    os.mkdir(parameterFilePath)
    os.mkdir(processedImagePath)
    os.mkdir(scaledImgPath)
    os.mkdir(sRecifiedImgPath)
    # -----------------------------------------------------------------------------
    #               >> Settings
    # -----------------------------------------------------------------------------
    # Save images with identified corners drawn 
    savePlot                 = True
    # Save scaled images 
    enableSaveScaledImgs     = False
    # Enable input image scaling (scale factor below)
    enableScaling            = False
    #=================================================================
    # Define calibration board
    #=================================================================
    # Define size of chessboard target.
    # [!!!] Number of inner corners per a chessboard row and column 
    #( patternSize = cvSize (points_per_row, points_per_colum) = 
    #cvSize(columns,rows) ).
    boardSize = (5,15)

    # Factor to calculate between diagonal distance between dots and vertical distance
    # between dots of the same column
    acircleFactor = math.sqrt(2)
    # Physical distance between pattern feater points [m]
    # > chessboard     -> horizontal/vertical spacing between corners
    # > assym. circles -> diagonal spacing between dots
    phys_corner_dist = 0.04 * acircleFactor
    
    sPatternType = "acircles"
    # -----------------------------------------------------------------------------
    #   >> Setup Logger
    # -----------------------------------------------------------------------------
    # Flag: Enable console prints 
    flagIsConsolePrint = True 
    # Flag: Create and save logging text file
    flagIsSaveLogFile  = True 
    log = PyLog(path, "CalibrationLog", flagIsConsolePrint, flagIsSaveLogFile)
    # -----------------------------------------------------------------------------
    # Empty folder for processed images 
    cleanFolder(processedImagePath)
    # Empty folder for calibration parameters:
    cleanFolder(parameterFilePath)
    # Empty folder for rectified images:
    cleanFolder(sRecifiedImgPath)
    # Empty folder for scaled images:
    cleanFolder(scaledImgPath)
    #------------------------------------------------------------------------------
    # >> Calibrate [monocular]
    #------------------------------------------------------------------------------
    # Initialize calibration
    calibrator = Calibrator(inputFilePath, 
                            processedImagePath,
                            scaledImgPath,
                            parameterFilePath,
                            sRecifiedImgPath,
                            boardSize,
                            phys_corner_dist,
                            sPatternType,
                            log)

    # Settings flags:
    calibrator.setEnableImgScaling(enableScaling)
    calibrator.setEnableMarkedImages(True)
    calibrator.setEnableSaveScaledImages(enableSaveScaledImgs)
    calibrator.setEnableRemapping(True)
    calibrator.setCropRectifiedImages(True)

    # Run [MONOCULAR CALIBRATION]
    calibrator.calibrate()

    # Calculate reprojection error per image
    calibrator.showReprojectionError()
    
    # Maximum allowable average reprojection error per image
    maxReprojThreshold = 0.013
    # Counter to track number of recalibration loops
    counter = 0
    # Maximum recalibration attempts to get the error down
    maxIter = 3
    # maximum average reprojection error per image in data set 
    maxError = 999
    
    while maxError > maxReprojThreshold and counter < maxIter :
        # Sort out outliers based on average recalibration error per image
        nrImagesDiscarded = calibrator.discardReprojectionOutliers(maxReprojThreshold)
        
        # If no images were discarded break the loop and stop recalibration
        if nrImagesDiscarded == 0:
            break
        
        # Recalibrate based on the revised image list 
        calibrator.recalibrate()
        
        # Calculate reprojection error per image
        calibrator.showReprojectionError()
        
        # Get maximum average recalibration error based on the recalibrated set
        listReprojError = calibrator.returnAveragReprojErrPerImage()
        maxError = max( listReprojError )
        
        # Update counter 
        counter = counter + 1

    # Save Calibration Results:
    calibrator.saveResults()

    # Produce rectified images
    calibrator.rectify()

    #------------------------------------------------------------------------------
    # >> Calibrate [stereo]
    #------------------------------------------------------------------------------
    # StereoImagePairDir = ""

    # RightCameraIdentifier = "right"
    # LeftCameraIdentifier  = "left"

    # imageFileExt          = "png"

    # calibrator.intializeStereoCalibration(StereoImagePairDir, 
    #                                       RightCameraIdentifier, 
    #                                       LeftCameraIdentifier, 
    #                                       imageFileExt)

    # calibrator.readImagePairs()

    # calibrator.stereoCalibrate()
    #------------------------------------------------------------------------------ 
    log.close()
    #------------------------------------------------------------------------------ 
    
if "__name__"==main():
    main()