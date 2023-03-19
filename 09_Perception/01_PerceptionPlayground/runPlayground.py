"""

Stereo bench playground 

This package allows to interface the stereo bench in different modes.
It's organized as in playground architecture allowing to enable/dissable 
each function.

"""
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import util.io_functions as io
import util.image_functions as img
import util.visualizers as show
import util.stereo_camera as camera
import util.ephemeris_interface as eph

from util.sun_finder import SunDetector
from util.imu_interface import ImuConnector 

from util.PyLog import PyLog
# -----------------------------------------------------------------------------------------------------------
#               [PLAY MODE]
# -----------------------------------------------------------------------------------------------------------
# Define allowable playmodes 
PM_STREAM_FROM_CAM              = 1
PM_STATIC_STEREO_TEST_FROM_IMG  = 2
PM_SUN_GAZING_TEST              = 3

# [!] Define play mode 
playMode = PM_SUN_GAZING_TEST

# Create and start streaming from stereo camera
# streamingMode options: rawimg, rectimg, dispmap, laplace
streamingMode = 'laplace'
# -----------------------------------------------------------------------------------------------------------
# [Define file paths]
sCalibrationMatricesFilePath  = os.path.join('assets','02_camera_calibration','stereo_calibration.yaml')
sLeftUndistortionMapFilePath  = os.path.join('assets','02_camera_calibration','caml_undistortion_map.tiff')
sRightUndistortionMapFilePath = os.path.join('assets','02_camera_calibration','camr_undistortion_map.tiff')

testIndex = 2
sTestImageFilePath = os.path.join('assets',
                                  '01_input_images','img_'+str(testIndex)+'.png')

# Enable laplacian transform
enableLaplacianTransform = True

# [Load camera calibration matrices]
cam_calibration = io.loadCalibrationParameters(sCalibrationMatricesFilePath)

# [Load undistortion maps]
undist_maps = io.loadStereoUndistortionMaps(sLeftUndistortionMapFilePath, sRightUndistortionMapFilePath)
# -----------------------------------------------------------------------------------------------------------
log = PyLog()

# [Load raw stereo images]
if playMode == PM_STREAM_FROM_CAM:
    camera.startStreaming(undist_maps, streamingMode)
elif playMode == PM_STATIC_STEREO_TEST_FROM_IMG :
    (imgl, imgr) = io.loadStereoImage(sTestImageFilePath)

    # [Rectify raw images]
    (rimgl, rimgr) = img.rectifyStereoImageSet(imgl, imgr, undist_maps)

    if enableLaplacianTransform:
        ddepth = cv.CV_8U
        kernel_size = 5
        rimgl = cv.GaussianBlur(rimgl,(5,5),0)
        rimgr = cv.GaussianBlur(rimgr,(5,5),0)

        rimgl = cv.Laplacian(rimgl, ddepth, ksize=kernel_size)
        rimgr = cv.Laplacian(rimgr, ddepth, ksize=kernel_size)

    # [Create Disparity Map]
    (rawDispMap, 
    dispMapFiltered, 
    dispMapFilteredAndNormalized) = img.createDisparityMap(rimgl, rimgr)

    io.saveArrayAsCsv(dispMapFiltered, "./output/filteredDisparityMap.csv")

    # [Create Depth Map]
    depthMap = img.computeDepthMapFromDispMap(dispMapFiltered, cam_calibration["P2"])

    io.saveArrayAsCsv(depthMap, "./output/filteredDepthMap.csv")

    # [Display images]
    enableSaveToFile = False

    enableShowRawAndRectified = False
    enableShowRectifiedOnly   = False
    enableShowDisparityMap    = True

    # Show raw and undistorted images
    if enableShowRawAndRectified:
        log.pLogMsg(' --> Show Rectified Stereo Image')
        show.plotStereoImageRectification(imgl, imgr, rimgl, rimgr, enableSaveToFile)

        log.pLogMsg(' --> Show Rectified Stereo Image')
        show.plotStereoImage(rimgl, rimgr)

    if enableShowDisparityMap:
        show.plotDisparityMap(rimgl, rimgr, rawDispMap, dispMapFiltered, enableSaveToFile)
elif playMode == PM_SUN_GAZING_TEST:
    log.pLogMsg('|                 [Detection Test]                 |')

    # Compute sun and gravity direction vectors in NED frame 
    sunVec_NED, gravVec_NED = eph.computeSunGravVecInNed()
    
    log.pLogMsg(f' - Sun direction vector     [NED]: {sunVec_NED}')
    log.pLogMsg(f' - Gravity direction vector [NED]: {gravVec_NED}')
    log.pLogMsg('')
    # # [IMU] Measure gravity direction vector in camera frame
    # imu = ImuConnector()

    # accVector_stdv_mss = [999, 999, 999]
    # iCounter    = 0
    # maxAttempts = 4
    # acc_std_thr = 0.03
    # while any(y > acc_std_thr for y in accVector_stdv_mss) and iCounter < maxAttempts:
    #     log.pLogMsg(f' Measurement attempt {iCounter}')
    #     numMeasurements = 10
    #     (accVector_mss, 
    #     accVector_average_mss, 
    #     accVector_stdv_mss,
    #     temp_average_deg) = imu.takePoseMeasurement(numMeasurements)
    #     iCounter = iCounter + 1
        
    # gravVec_CAM = accVector_average_mss
    # log.pLogMsg(f' - Gravity direction vector [CAM]: {gravVec_CAM}')
    # log.pLogMsg(f' - Gravity vector stdv      [CAM]: {accVector_stdv_mss}')
    # log.pLogMsg(f' - IMU temperature         [degC]: {temp_average_deg}')
    # log.pLogMsg('')
    isLoadFromFile = True
    
    if not isLoadFromFile:
        # Capture stereo image pair 
        (imgl, imgr) = camera.captureStereoImagePair()
    else:
        (imgl, imgr) = io.loadStereoImage(sTestImageFilePath)
        
    # [Rectify raw images]
    (rimgl, rimgr) = img.rectifyStereoImageSet(imgl, imgr, undist_maps)
        
    # [CAMERA] Measure sun direction in camera frame
    sFinder = SunDetector()
    sFinder.detect(rimgl, log)
    
    cv.imwrite('./output/test.png', sFinder.debugImg)

else:
    log.pLogErr('playMode not valid. Exiting')
    exit(1)
# -----------------------------------------------------------------------------------------------------------