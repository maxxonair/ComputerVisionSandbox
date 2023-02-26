import os
import matplotlib.pyplot as plt
import numpy as np

import util.io_functions as io
import util.image_functions as img
import util.visualizers as show
import util.stereo_camera as camera
# -----------------------------------------------------------------------------------------------------------
# [Define file paths]
sCalibrationMatricesFilePath  = os.path.join('assets','02_camera_calibration','stereo_calibration.yaml')
sLeftUndistortionMapFilePath  = os.path.join('assets','02_camera_calibration','caml_undistortion_map.tiff')
sRightUndistortionMapFilePath = os.path.join('assets','02_camera_calibration','camr_undistortion_map.tiff')

testIndex = 8
sTestImageFilePath = os.path.join('assets',
                                  '01_input_images','img_'+str(testIndex)+'.png')

# Enable live stream from the stereo camera. 
# If set False -> load and process test image from file instead 
enableCameraLiveStream = False

# [Load camera calibration matrices]
cam_calibration = io.loadCalibrationParameters(sCalibrationMatricesFilePath)

# [Load undistortion maps]
undist_maps = io.loadStereoUndistortionMaps(sLeftUndistortionMapFilePath, sRightUndistortionMapFilePath)
# -----------------------------------------------------------------------------------------------------------
# [Load raw stereo images]
if not enableCameraLiveStream:
    (imgl, imgr) = io.loadStereoImage(sTestImageFilePath)

    # [Rectify raw images]
    (rimgl, rimgr) = img.rectifyStereoImageSet(imgl, imgr, undist_maps)

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
        print(' --> Show Rectified Stereo Image')
        show.plotStereoImageRectification(imgl, imgr, rimgl, rimgr, enableSaveToFile)

    if enableShowRectifiedOnly:
        print(' --> Show Rectified Stereo Image')
        show.plotStereoImage(rimgl, rimgr)

    if enableShowDisparityMap:
        show.plotDisparityMap(rimgl, rimgr, dispMapFiltered, depthMap, enableSaveToFile)
else:
    # Create and start streaming from stereo camera
    # streamingMode options: rawimg, rectimg, dispmap
    streamingMode = 'rawimg'
    camera.startStreaming(undist_maps, streamingMode)
# -----------------------------------------------------------------------------------------------------------