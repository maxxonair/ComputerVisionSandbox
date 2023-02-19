
import os
import matplotlib.pyplot as plt
import numpy as np

import util.io_functions as io
import util.image_functions as img
import util.visualizers as show

# [Define file paths]
sCalibrationMatricesFilePath  = os.path.join('assets','02_camera_calibration','stereo_calibration.yaml')
sLeftUndistortionMapFilePath  = os.path.join('assets','02_camera_calibration','caml_undistortion_map.tiff')
sRightUndistortionMapFilePath = os.path.join('assets','02_camera_calibration','camr_undistortion_map.tiff')

testIndex = 26
sTestImageFilePath=os.path.join('assets','01_input_images','img_'+str(testIndex)+'.png')

# [Load camera calibration matrices]
cam_calibration = io.loadCalibrationParameters(sCalibrationMatricesFilePath)

# [Load undistortion maps]
undist_maps = io.loadStereoUndistortionMaps(sLeftUndistortionMapFilePath, sRightUndistortionMapFilePath)

# [Load raw stereo images]
(imgl, imgr) = io.loadStereoImage(sTestImageFilePath)

# [Rectify raw images]
(rimgl, rimgr) = img.rectifyStereoImageSet(imgl, imgr, undist_maps)

# [Display images]
# Show raw and undistorted images
print(' --> Show Rectified Stereo Image')
show.plotStereoImageRectification(imgl, imgr, rimgl, rimgr)

# print()
# print(' --> Show Rectified Stereo Image')
# show.plotStereoImage(rimgl, rimgr)