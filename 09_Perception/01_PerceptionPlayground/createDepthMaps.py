
import cv2 as cv
import numpy as np
from pathlib import Path
import json
import sys, os
from os import mkdir
from tqdm import tqdm
import glob
import rerun as rr

import util.io_functions as io
import util.image_functions as imgf

from util.PyLog import PyLog


# -------- DEFINE CONSTANTS
# Path to stereo camera calibration root folder 
calibrationDataPath = "/Users/mrx/Documents/003_Tools/42_ImageProcessing/07_CameraCalibration/02_camera_calibration_results/2024_04_01__15_43_13_camera_01_calibration"

# Path to test image set directory
testImageSetPath = '/Users/mrx/Documents/003_Tools/42_ImageProcessing/09_Perception/01_PerceptionPlayground/output/2024_04_14_10_01_08_frame_capture'

log = PyLog()

# Calibration mode: Raw images are resized to calibration size 
# Note: This mode must be used by streaming calls that rectify the image
#       downstream!
SHOW_MODE_CALIBRATION   = 5

ENABLE_RERUN = True

def main():
  if ENABLE_RERUN:
    rr.init("rerun_example_depth_image_3d", spawn=True)
    
  # ---- (1) Read calibration data 
  
  # [Load camera calibration matrices]
  sCamCalibFilePath = (Path(calibrationDataPath) 
                                  / '03_camera_parameters' 
                                  / 'stereo_calibration.yaml')
  sLeftUndistortionMapFilePath = (Path(calibrationDataPath) 
                                  / '03_camera_parameters' 
                                  / 'caml_undistortion_map.tiff')
  sRightUndistortionMapFilePath = (Path(calibrationDataPath) 
                                  / '03_camera_parameters' 
                                  / 'camr_undistortion_map.tiff')
  
  if not sCamCalibFilePath.exists():
    log.pLogErr(f'CALIBRATION FILE NOT FOUND. -> {sCamCalibFilePath.absolute().as_posix()}')
    log.pLogErr(f'Exiting.')
    log.close()
    exit(1)

  if not sLeftUndistortionMapFilePath.exists():
    log.pLogErr(f'LEFT DISTORTION CORRECTION MAP NOT FOUND. -> {sLeftUndistortionMapFilePath.absolute().as_posix()}')
    log.pLogErr(f'Exiting.')
    log.close()
    exit(1)
    
  if not sRightUndistortionMapFilePath.exists():
    log.pLogErr(f'RIGHT DISTORTION CORRECTION MAP NOT FOUND. -> {sRightUndistortionMapFilePath.absolute().as_posix()}')
    log.pLogErr(f'Exiting.')
    log.close()
    exit(1)
    
  log.pLogMsg('    [Load stereo camera calibration parameters]')
  cam_calibration = io.loadCalibrationParameters(
    sCamCalibFilePath.absolute().as_posix(), log)
  
  focalLength_px = cam_calibration['P2'][0,0]
  
  # [Load undistortion maps]
  log.pLogMsg('    [Load stereo camera undistortion maps]')
  undist_maps = io.loadStereoUndistortionMaps(
    sLeftUndistortionMapFilePath.absolute().as_posix(),
    sRightUndistortionMapFilePath.absolute().as_posix(),
    log)

  # ---- (2) Read image pairs
  # -- (2a) Read image set meta data json 
  metaFilePath = Path(testImageSetPath) / 'stereo_img_meta.json' 
  
  # Check if file exist at given path
  if not metaFilePath.exists():
    log.pLogErr(f'IMAGE SET META DATA FILE NOT FOUND. -> {metaFilePath.absolute().as_posix()}')
    log.pLogErr(f'Exiting.')
    log.close()
    exit(1)
    
  # Load image set meta data to dictionary 
  with open(metaFilePath.absolute().as_posix()) as json_file:
    metadata = json.load(json_file)

  stereoImgList    = glob.glob((Path(testImageSetPath) 
                                / f"{metadata['img_prefix']}*.png").absolute().as_posix())
  stereoImgList.sort()
  
  if len(stereoImgList) != metadata['number_of_images']:
    log.pLogWrn(("Number of images found in directory does not match the number "
                f"given in the meta data file {len(stereoImgList)} vs {metadata['number_of_images']}"))

  if metadata['mode'] is not SHOW_MODE_CALIBRATION:
    log.pLogErr(f'Test images were not taken in calibration mode (reported mode {metadata['mode']})')
    log.pLogErr(f'Capture mode is not valid for this processing. Exiting.')
    log.close()
    exit(1)
    
  # Create sub-directories to save disparity and depth maps to, if not already
  # existent 
  if not (Path(testImageSetPath) / '01_disparity_maps' ).exists():
    mkdir((Path(testImageSetPath) / '01_disparity_maps' ).absolute().as_posix())  
  if not (Path(testImageSetPath) / '02_depth_maps' ).exists():
    mkdir((Path(testImageSetPath) / '02_depth_maps' ).absolute().as_posix())  
  
  for imgIndx, stereoImgFilePath in tqdm(enumerate(stereoImgList)):
    img = cv.imread(stereoImgFilePath)

    if img is None: 
        log.pLogErr(f'Image loading returned: None')
        log.pLogErr(f'Image path: {stereoImgFilePath}')
        exit(1)

    # If input is not grayscale -> Convert image to greyscale
    if(len(img.shape)<3):
        gray = img
    else:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # --- Extract cropping coordinates from meta data file 
    # Height of the concatenated stereo iamge and the split single (left & right)
    # image in pixel 
    imgHeight = metadata['stereo_img_res'][0]
    # Width of the concatenated stereo image in pixel 
    stereoImgWidth = metadata['stereo_img_res'][1]
    # Width of the split left & right image in pixel 
    imgWidth = metadata['y_cut_coord']
    
    # Create left & right image as separate arrays
    imgl  = gray[ 0:imgHeight , 0:imgWidth ]
    imgr = gray[ 0:imgHeight , imgWidth:stereoImgWidth ]
    
    # Rectify left and right images
    (rimgl, rimgr) = imgf.rectify_stereo_image(imgl, imgr, undist_maps )
    
    # Create disparity maps
    (displ, filteredImg, filteredImgNormalized) = imgf.create_disparity_map(rimgl, rimgr)
    
    # Create depth maps 
    depthMap = imgf.compute_depth_map_from_disp_map(filteredImg, cam_calibration['P2'])
    
    # Save created products to file 
    np.savetxt((Path(testImageSetPath) / '01_disparity_maps' / f'disp_map_{imgIndx}').absolute().as_posix(),
               np.asarray(filteredImg),
               delimiter=",")
    np.savetxt((Path(testImageSetPath) / '02_depth_maps' / f'depth_map_{imgIndx}').absolute().as_posix(),
               np.asarray(depthMap),
               delimiter=",")
    
    if ENABLE_RERUN:
      # If we log a pinhole camera model, the depth gets automatically back-projected to 3D
      rr.log(
          "world/camera",
          rr.Pinhole(
              width=depthMap.shape[1],
              height=depthMap.shape[0],
              focal_length=200,
          ),
      )

      # Log the tensor.
      rr.log("world/camera/depth", rr.DepthImage(depthMap, meter=1.0))
    
if __name__=="__main__":
  main()