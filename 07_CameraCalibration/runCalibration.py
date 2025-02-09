#!/usr/bin/env python3
# -------------------------------------------------------------------------
#                      >> Camera calibration <<
# -------------------------------------------------------------------------
#
#   Function to support mono and stereo camera calibration
#
# -------------------------------------------------------------------------
import os
import datetime
import argparse
from pathlib import Path

from util.supportFunctions import cleanFolder
from util.PyLog import PyLog
from util.MonoCalibrator import MonoCalibrator
from util.StereoCalibrator import StereoCalibrator

import camera_calibration_targets.chessboard.chessboard_a3 as chessboard
import camera_calibration_targets.asymmetric_circles.acircles_a3 as acircles


def main(mode: str,
         img_path: Path,
         calib_board: str,
         enable_remapping: bool = True,
         save_corner_detect_debug: bool = True,
         enable_scaling: bool = False,
         use_blob_detector: bool = False,
         crop_rect_imgs: bool = False):
  """


  """
  # --------------------------------------------------------------------------
  #               >> Folder Setup
  # --------------------------------------------------------------------------
  camera_ID = "01"

  # >> Setup folder structure
  # Setup folder to save images to
  dateTimeStr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
  folder_name = dateTimeStr + "_camera_" + str(camera_ID) + "_calibration"
  # start time counter
  path = "02_camera_calibration_results/" + folder_name
  os.makedirs(path)

  # Set file paths:
  parameterFilePath = os.path.join(path, "03_camera_parameters/")
  processedImagePath = os.path.join(path, "02_processed_calibration_images/")
  inputFilePath = os.path.join(Path(img_path).absolute().as_posix())
  scaledImgPath = os.path.join(path, "06_scaled_images/")
  sRecifiedImgPath = os.path.join(path, '05_corrected_images/')
  sDisparityMapsPath = os.path.join(path, '07_disparity_map/')

  os.makedirs(parameterFilePath)
  os.makedirs(processedImagePath)
  os.makedirs(scaledImgPath)
  os.makedirs(sRecifiedImgPath)
  os.makedirs(sDisparityMapsPath)
  # =================================================================
  # Define calibration board
  # =================================================================
  # Define size of chessboard target.

  if calib_board == 'circles':
    aCircBoardOne = acircles.getCalibrationBoardProperties()
  elif calib_board == 'chess':
    chessBoardOne = chessboard.getCalibrationBoardProperties()
  else:
    raise RuntimeError(
        f'Calibration board invalid ({calib_board}). Select circles or chess')

  # Set board dimsension in corner points per column and row
  # E.g. the following chessboard, boardSize = (5,6)
  # --->
  # | [x][ ][x][ ][x][ ][x]
  # | [ ][x][ ][x][ ][x][ ]
  # | [x][ ][x][ ][x][ ][x]
  # | [ ][x][ ][x][ ][x][ ]
  # | [x][ ][x][ ][x][ ][x]
  # | [ ][x][ ][x][ ][x][ ]
  # |
  # v

  boardSize = chessBoardOne["boardSize"]

  # Physical distance between pattern feature points [m]
  # > chessboard     -> horizontal/vertical spacing between corners
  # > assym. circles -> diagonal spacing between dots
  phys_corner_dist = chessBoardOne["cornerDist_m"]

  # Set pattern type:
  # Options: chessboard, acircles
  sPatternType = chessBoardOne["PatternType"]
  # -----------------------------------------------------------------------
  #   >> Setup Logger
  # -----------------------------------------------------------------------
  # Flag: Enable console prints
  flagIsConsolePrint = True
  # Flag: Create and save logging text file
  flagIsSaveLogFile = True
  log = PyLog(path,
              "CalibrationLog",
              flagIsConsolePrint,
              flagIsSaveLogFile)
  
  # -----------------------------------------------------------------------
  # Empty folder for processed images
  cleanFolder(processedImagePath)
  # Empty folder for calibration parameters:
  cleanFolder(parameterFilePath)
  # Empty folder for rectified images:
  cleanFolder(sRecifiedImgPath)
  # Empty folder for scaled images:
  cleanFolder(scaledImgPath)
  # -----------------------------------------------------------------------
  # >> Calibrate [MONOCULAR]
  # -----------------------------------------------------------------------
  if mode == "mono":
    # Initialize calibration
    calibrator = MonoCalibrator(inputFilePath,
                                processedImagePath,
                                scaledImgPath,
                                parameterFilePath,
                                sRecifiedImgPath,
                                boardSize,
                                phys_corner_dist,
                                sPatternType,
                                log)

    # Settings flags:
    calibrator.setEnableImgScaling(enable_scaling)
    calibrator.setEnableSaveScaledImages(enable_scaling)
    calibrator.setEnableMarkedImages(save_corner_detect_debug)
    calibrator.setEnableRemapping(enable_remapping)
    calibrator.setCropRectifiedImages(crop_rect_imgs)

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

    while maxError > maxReprojThreshold and counter < maxIter:
      # Sort out outliers based on average recalibration error per image
      nrImagesDiscarded = calibrator.discardReprojectionOutliers(
          maxReprojThreshold)

      # If no images were discarded break the loop and stop recalibration
      if nrImagesDiscarded == 0:
        break

      # Recalibrate based on the revised image list
      calibrator.recalibrate()

      # Calculate reprojection error per image
      calibrator.showReprojectionError()

      # Get maximum average recalibration error based on the recalibrated set
      listReprojError = calibrator.returnAveragReprojErrPerImage()
      maxError = max(listReprojError)

      # Update counter
      counter = counter + 1

    # Save Calibration Results:
    calibrator.saveResults()

    # Produce rectified images
    calibrator.rectify()

  # -----------------------------------------------------------------------
  # >> Calibrate [STEREO]
  # -----------------------------------------------------------------------
  if mode == "stereo":
    # Initialize stereo calibration instance
    calibrator = StereoCalibrator(inputFilePath,
                                  processedImagePath,
                                  scaledImgPath,
                                  parameterFilePath,
                                  sRecifiedImgPath,
                                  sDisparityMapsPath,
                                  boardSize,
                                  phys_corner_dist,
                                  sPatternType,
                                  log)

    # ----- SETTING FLAGS -----
    calibrator.setEnableImgScaling(enable_scaling)
    # Enable saving raw images with corner points drawn onto them
    calibrator.setEnableMarkedImages(save_corner_detect_debug)
    #
    calibrator.setEnableSaveScaledImages(enable_scaling)
    # This setting does not work with rectifyCalibImages
    # -> (concat differently sized images ...)
    calibrator.setCropRectifiedImages(crop_rect_imgs)
    # Blob detector is only relevant when calibration with a circle pattern board
    # Using blob detector on top of pattern recognition improves the yield
    calibrator.setEnableUseBlobDetector(use_blob_detector)

    # Run >> read stereo pairs + [MONOCULAR CALIBRATION]
    calibrator.readStereoPairs()

    # [CALIBRATE stereo cameras]
    calibrator.stereoCalibrate()

    # Rectify calibration image set and create disparity maps ()
    calibrator.rectifyCalibImages()

    # Attempt to discard 2 sigma outliers
    calibrator.discardReprojectionOutliers()
    calibrator.recalibrate()

    # Draw re-projected image points on rectified images to check calibration
    # calibrator.drawReprojectedCornerPoints()

  # -----------------------------------------------------------------------
  log.close()
  # -----------------------------------------------------------------------


if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      prog='[CameraCalibration]',
      description='Camera calibration software supporting mono as well as stereo camera calibration')

  parser.add_argument('-m',
                      '--mode',
                      required=True,
                      help="Calibration mode: select mono or stereo")
  parser.add_argument('-b',
                      '--board',
                      required=True,
                      help="Select calibration board: chess or cicles")
  parser.add_argument('-p',
                      '--path',
                      required=True,
                      help="Path to the calibration image directory")

  args = parser.parse_args()

  img_path = Path(args.path)
  if not img_path.exists():
    raise RuntimeError(f"Input image directory not found: {img_path}")

  if not (args.mode == 'mono' or args.mode == 'stereo'):
    raise RuntimeError('Calibration mode invalid. Select mono or stereo')

  if not (args.board == 'chess' or args.board == 'circles'):
    raise RuntimeError(
        'Calibration board setting invalid. Select chess or circles')

  main(args.mode, img_path, args.board)
