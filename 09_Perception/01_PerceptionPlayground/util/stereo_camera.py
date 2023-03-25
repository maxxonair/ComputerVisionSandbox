
import cv2 as cv
import os
from time import perf_counter
import datetime
import numpy as np
import matplotlib.pyplot as plt

import util.image_functions as img

class StereoCamera:
  
  # Define imageShow window title:
  imgWindowName = "StereoBench camera image feed [.mk0]"

  # Streaming mode constants
  SHOW_MODE_DISPARITY_MAP = 1
  SHOW_MODE_RECTIFIED     = 2
  SHOW_MODE_RAW_IMG       = 3
  SHOW_MODE_LAPLCE_IMG    = 4
  
  def __init__(self, log):
    self.log = log
    self.imgl = []
    self.imgr = []
    self.stereoImgGray = []

  def captureStereoImagePair(self):
    # [!] Left stereo bench camera port ID
    cam01_port = 0
    # [!] Rigth stereo bench camera port ID
    cam02_port = 1

    self.log.pLogMsg(' [Open camera interfaces]')
    self.log.pLogMsg('')
    # Open Video capture for both cameras
    leftCamInterface  = cv.VideoCapture(cam01_port)
    rightCamInterface = cv.VideoCapture(cam02_port)

    # Check if both webcams have been opened correctly
    if not leftCamInterface.isOpened():
      raise IOError("Cannot open webcam 01")
    if not rightCamInterface.isOpened():
      raise IOError("Cannot open webcam 02")
    suc1 = False
    suc2 = False
    maxAttempts = 20
    attemptCount = 0 
    while (not suc1 and not suc2 or attemptCount < maxAttempts ):
      (suc1, 
       self.imgl, 
       suc2, 
       self.imgr, 
       timetag_ms) = self._acquireStereoImagePair(leftCamInterface, 
                                                  rightCamInterface, 
                                                  False)
      attemptCount = attemptCount + 1

    if suc1 and suc2: 
      self.log.pLogMsg('[x] Stereo frame captured successfully.')
      return self.imgl, self.imgr
    else:
      self.log.pLogMsg(' > ERROR < Image capture failed.')
      self.imgl = []
      self.imgr = []
      return self.imgl, self.imgr


  def startStreaming(self, undist_maps, showProduct, showImages=False):
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
      showMode   = self.SHOW_MODE_RAW_IMG
    elif showProduct == 'rectimg':
      showMode   = self.SHOW_MODE_RECTIFIED
    elif showProduct == 'dispmap':
      showMode   = self.SHOW_MODE_DISPARITY_MAP
    elif showProduct == 'laplace':
      showMode   = self.SHOW_MODE_LAPLCE_IMG
    else:
      self.log.pLogMsg('ERROR: showProduct not recognized. Using default show raw images.')
      showMode   = self.SHOW_MODE_RAW_IMG
    
    # [!] Left stereo bench camera port ID
    cam01_port = 0
    # [!] Rigth stereo bench camera port ID
    cam02_port = 1
    # Initialise image index for saved files 
    img_index  = 0

    self.log.pLogMsg(' [Open camera interfaces]')
    self.log.pLogMsg('')
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
          self.log.pLogMsg('')
          self.log.pLogMsg(' ==> [Start Streaming]')
          self.log.pLogMsg('')

        (suc1, 
         self.imgl, 
         suc2, 
         self.imgr, 
         timetag_ms) = self._acquireStereoImagePair(leftCamInterface, 
                                                    rightCamInterface, 
                                                    isPromptGrabDelay)
        grabDiffList.append(timetag_ms)

        if suc1 and suc2: 
          # Process and display image pair depending on selected mode
          if showMode == self.SHOW_MODE_DISPARITY_MAP:
            # Rectify camera images 
            (rimgl, rimgr) = img.rectifyStereoImageSet(self.imgl, self.imgr, undist_maps)
            (rawDispMap, 
            dispMapFiltered, 
            dispMapFilteredAndNormalized) = img.createDisparityMap(rimgl, rimgr)
            self.imgToShow = dispMapFilteredAndNormalized
            # self.imgToShow = cv.applyColorMap(self.imgToShow, cv.COLORMAP_JET)
          elif showMode == self.SHOW_MODE_RECTIFIED:\
            # Rectify camera images 
            (rimgl, rimgr) = img.rectifyStereoImageSet(self.imgl, self.imgr, undist_maps)
            self.imgToShow = cv.hconcat([rimgl, rimgr])
          elif showMode == self.SHOW_MODE_RAW_IMG:
            self.imgToShow = cv.hconcat([self.imgl, self.imgr])
            self.imgToShow = cv.resize(self.imgToShow, (1200,600), interpolation=cv.INTER_AREA)
            self.stereoImgGray = cv.cvtColor(self.imgToShow, cv.COLOR_BGR2GRAY) 
          elif showMode == self.SHOW_MODE_LAPLCE_IMG:
            kernel_size = 5
            # Rectify camera images 
            (rimgl, rimgr) = img.rectifyStereoImageSet(self.imgl, self.imgr, undist_maps)
            # Laplace transform rectified image pair 
            (rimgl, rimgr) = self._laplaceTransformStereoPair(rimgl, rimgr, kernel_size)
            # Show image
            self.imgToShow = cv.hconcat([rimgl, rimgr])
          else:
            self.log.pLogMsg("[ERR] Error showMode not valid. Exiting")
            break

          # Show image pair 
          if showImages:
            cv.imshow(self.imgWindowName, self.imgToShow)
        else:
          self.log.pLogMsg("Image retrieval failed.")
            
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
          self.log.pLogMsg("Save image to file: "+img_file_path)
          self.log.pLogMsg(f"Image dimensions: {(self.imgToShow).shape}")
          cv.imwrite(img_file_path, self.imgToShow) 
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
          self.log.pLogMsg("Save statistics to file: "+csv_full_file_path)
          # Save complete grabDiffList as csv
          np.savetxt(csv_full_file_path, grabDiffList, delimiter=",")
          # Save histogram as png
          self._createGrabTimingHistogram(grabDiffList, png_full_file_path)

    leftCamInterface.release()
    rightCamInterface.release()
    cv.destroyAllWindows()

  def _acquireStereoImagePair(self, leftCamInterface, rightCamInterface, isPromptGrabDelay):
      # Grab frames from both cameras
      t_start = perf_counter()
      ret1 = leftCamInterface.grab()
      t_end = perf_counter()
      ret2 = rightCamInterface.grab()
      timetag_ms = ( t_end - t_start ) * 1000

      suc1 = []
      suc2 = []
      self.imgl = []
      self.imgr = []

      # Print time taken to grab one cameras image
      if isPromptGrabDelay:
        self.log.pLogMsg("Grab diff [ms] : "+str(timetag_ms))
      if not ret1 and not ret2:
        self.log.pLogMsg("Failed to grab frames")  
        if not ret1 :
          self.log.pLogMsg("Failed to grab camera 1 frame")
        if not ret2 :
          self.log.pLogMsg("Failed to grab camera 2 frame")
        # TODO: Add action here       
      else:
        # Read camera frames
        suc1, self.imgl = leftCamInterface.retrieve()
        suc2, self.imgr = rightCamInterface.retrieve()

      return (suc1,self. imgl, suc2, self.imgr, timetag_ms)

  def _createGrabTimingHistogram(self, grabDiffList, pngFilePath):
    plt.hist(grabDiffList, bins=60, range=[0, 60], color = "gray")
    plt.title('Stereo camera session: Grab frame difference')
    plt.xlabel('Time between left and right frame [ms]')
    plt.savefig(pngFilePath)

  def _laplaceTransformStereoPair(self, limg, rimg, kernel_size):
    ddepth = cv.CV_8U
    limg = cv.Laplacian(limg, ddepth, ksize=kernel_size)
    rimg = cv.Laplacian(rimg, ddepth, ksize=kernel_size)
    return limg, rimg
