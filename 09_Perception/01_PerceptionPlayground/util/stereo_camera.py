
import cv2 as cv
import os
from time import perf_counter
import datetime
import numpy as np
import matplotlib.pyplot as plt

import util.image_functions as img

# Define imageShow window title:
imgWindowName = "StereoBench camera image feed [.mk0]"

# Streaming mode constants
SHOW_MODE_DISPARITY_MAP = 1
SHOW_MODE_RECTIFIED     = 2
SHOW_MODE_RAW_IMG       = 3
SHOW_MODE_LAPLCE_IMG    = 4

def captureStereoImagePair():
  # [!] Left stereo bench camera port ID
  cam01_port = 0
  # [!] Rigth stereo bench camera port ID
  cam02_port = 1

  print(' [Open camera interfaces]')
  print()
  # Open Video capture for both cameras
  leftCamInterface  = cv.VideoCapture(cam01_port)
  rightCamInterface = cv.VideoCapture(cam02_port)

  # Check if both webcams have been opened correctly
  if not leftCamInterface.isOpened():
    raise IOError("Cannot open webcam 01")
  if not rightCamInterface.isOpened():
    raise IOError("Cannot open webcam 02")
  
  (suc1, imgl, suc2, imgr, timetag_ms) = _acquireStereoImagePair(leftCamInterface, 
                                                                rightCamInterface, 
                                                                False)

  if suc1 and suc2: 
    return imgl, imgr
  else:
    print('Image capture failed.')
    imgl = []
    imgr = []
    return imgl, imgr


def startStreaming(undist_maps, showProduct):
  # Setup folder to save images to 
  dateTimeStr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  folder_name = dateTimeStr+"_frame_capture"
  # start time counter 
  path = os.path.join("output",folder_name)
  # Internal status flag to track if folder to save images to has been set
  # up already 
  isFolderSetup     = False
  # Internal flag to track if start of streaming message has been displayed
  # This message should only be shown at start up 
  isStartupMsgShown = False

  # Control flag to show or hide delay between left and right image grab in 
  # terminal 
  # TODO: Move this somewhere sensible
  isPromptGrabDelay = False

  grabDiffList = []

  if showProduct == 'rawimg':
    showMode   = SHOW_MODE_RAW_IMG
  elif showProduct == 'rectimg':
    showMode   = SHOW_MODE_RECTIFIED
  elif showProduct == 'dispmap':
    showMode   = SHOW_MODE_DISPARITY_MAP
  elif showProduct == 'laplace':
    showMode   = SHOW_MODE_LAPLCE_IMG
  else:
    print('ERROR: showProduct not recognized. Using default show raw images.')
    showMode   = SHOW_MODE_RAW_IMG
  
  # [!] Left stereo bench camera port ID
  cam01_port = 0
  # [!] Rigth stereo bench camera port ID
  cam02_port = 1
  # Initialise image index for saved files 
  img_index  = 0

  print(' [Open camera interfaces]')
  print()
  # Open Video capture for both cameras
  leftCamInterface  = cv.VideoCapture(cam01_port)
  rightCamInterface = cv.VideoCapture(cam02_port)

  # Check if both webcams have been opened correctly
  if not leftCamInterface.isOpened():
    raise IOError("Cannot open webcam 01")
  if not rightCamInterface.isOpened():
    raise IOError("Cannot open webcam 02")

  while True:
      if not isStartupMsgShown:
        isStartupMsgShown = True
        print()
        print(' ==> [Start Streaming]')
        print()

      (suc1, imgl, suc2, imgr, timetag_ms) = _acquireStereoImagePair(leftCamInterface, 
                                                                    rightCamInterface, 
                                                                    isPromptGrabDelay)
      grabDiffList.append(timetag_ms)

      if suc1 and suc2: 
        
        # Process and display image pair depending on selected mode
        if showMode == SHOW_MODE_DISPARITY_MAP:
          # Rectify camera images 
          (rimgl, rimgr) = img.rectifyStereoImageSet(imgl, imgr, undist_maps)
          (rawDispMap, 
          dispMapFiltered, 
          dispMapFilteredAndNormalized) = img.createDisparityMap(rimgl, rimgr)
          imgToShow = dispMapFilteredAndNormalized
          # imgToShow = cv.applyColorMap(imgToShow, cv.COLORMAP_JET)
        elif showMode == SHOW_MODE_RECTIFIED:\
          # Rectify camera images 
          (rimgl, rimgr) = img.rectifyStereoImageSet(imgl, imgr, undist_maps)
          imgToShow = cv.hconcat([rimgl, rimgr])
        elif showMode == SHOW_MODE_RAW_IMG:
          imgToShow = cv.hconcat([imgl, imgr])
          imgToShow = cv.resize(imgToShow, (1200,600), interpolation=cv.INTER_AREA)
        elif showMode == SHOW_MODE_LAPLCE_IMG:
          kernel_size = 5
          # Rectify camera images 
          (rimgl, rimgr) = img.rectifyStereoImageSet(imgl, imgr, undist_maps)
          # Laplace transform rectified image pair 
          (rimgl, rimgr) = _laplaceTransformStereoPair(rimgl, rimgr, kernel_size)
          # Show image
          imgToShow = cv.hconcat([rimgl, rimgr])
        else:
          print("[ERR] Error showMode not valid. Exiting")
          break

        # Show image pair 
        cv.imshow(imgWindowName, imgToShow)
      else:
        print("Image retrieval failed.")
          
      # Wait for key inputs
      key = cv.waitKey(1)
      # Define key input actions
      if key == ord('q'):
        # [q] -> exit
        break
      elif key == 27:
        # [ESC] -> exit
        break
      elif key == ord('c'):
        # [c] -> Save image to file
        if not isFolderSetup:
          # Create time tagged folder to save image pairs to
          os.mkdir(path)
          isFolderSetup = True
        # Define image name 
        img_name = "img_"+str(img_index)+".png"
        # Define complete image save path 
        img_file_path = os.path.join(path,img_name)
        print("Save image to file: "+img_file_path)
        cv.imwrite(img_file_path, imgToShow) 
        img_index = img_index + 1
      elif key == ord('s'):
        # [s] -> Save statistics
        if not isFolderSetup:
          # Create time tagged folder to save image pairs to
          os.mkdir(path)
          isFolderSetup = True
        # Define image name 
        csv_file_name = "grab_timing_statistics.csv"
        # Define complete image save path 
        csv_full_file_path = os.path.join(path,csv_file_name)
        png_full_file_path = os.path.join(path,'grab_timing_statistics.png')
        print("Save statistics to file: "+csv_full_file_path)
        # Save complete grabDiffList as csv
        np.savetxt(csv_full_file_path, grabDiffList, delimiter=",")
        # Save histogram as png
        _createGrabTimingHistogram(grabDiffList, png_full_file_path)

  leftCamInterface.release()
  rightCamInterface.release()
  cv.destroyAllWindows()

def _acquireStereoImagePair(leftCamInterface, rightCamInterface, isPromptGrabDelay):
    # Grab frames from both cameras
    t_start = perf_counter()
    ret1 = leftCamInterface.grab()
    t_end = perf_counter()
    ret2 = rightCamInterface.grab()
    timetag_ms = ( t_end - t_start ) * 1000

    suc1 = []
    suc2 = []
    imgl = []
    imgr = []

    # Print time taken to grab one cameras image
    if isPromptGrabDelay:
      print("Grab diff [ms] : "+str(timetag_ms))
    if not ret1 and not ret2:
      print("Failed to grab frames")  
      if not ret1 :
        print("Failed to grab camera 1 frame")
      if not ret2 :
        print("Failed to grab camera 2 frame")
      # TODO: Add action here       
    else:
      # Read camera frames
      suc1, imgl = leftCamInterface.retrieve()
      suc2, imgr = rightCamInterface.retrieve()

    return (suc1, imgl, suc2, imgr, timetag_ms)

def _createGrabTimingHistogram(grabDiffList, pngFilePath):
  plt.hist(grabDiffList, bins=60, range=[0, 60], color = "gray")
  plt.title('Stereo camera session: Grab frame difference')
  plt.xlabel('Time between left and right frame [ms]')
  plt.savefig(pngFilePath)

def _laplaceTransformStereoPair(limg, rimg, kernel_size):
  ddepth = cv.CV_8U
  limg = cv.Laplacian(limg, ddepth, ksize=kernel_size)
  rimg = cv.Laplacian(rimg, ddepth, ksize=kernel_size)
  return limg, rimg
