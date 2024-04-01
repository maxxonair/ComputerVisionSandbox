
import cv2 as cv
import os
from time import perf_counter
import datetime
import numpy as np
import matplotlib.pyplot as plt

import util.image_functions as img
import util.constants as cnst

class StereoCamera:
  
  # Define imageShow window title:
  imgWindowName = "StereoBench stream [.mk0]"

  # Streaming mode constants
  SHOW_MODE_DISPARITY_MAP = 1
  SHOW_MODE_RECTIFIED     = 2
  SHOW_MODE_RAW_IMG       = 3
  SHOW_MODE_LAPLCE_IMG    = 4
  # Calibration mode: Raw images are resized to calibration size 
  # Note: This mode must be used by streaming calls that rectify the image
  #       downstream!
  SHOW_MODE_CALIBRATION   = 5

  MAX_CONNECTION_ATTEMPTS = 20

  # [!] Left and right stereo bench camera port ID
  LEFT_CAMERA_PORT  = 0
  RIGHT_CAMERA_PORT = 1
  
  # Correct for any camera rotation
  # If true -> Rotate image by 180 degree
  ENABLE_ROTATE_LEFT_IMG  = True
  ENABLE_ROTATE_RIGHT_IMG = False

  def __init__(self, log):
    self.log = log
    self.imgl = []
    self.imgr = []
    self.stereoImgGray = []

  def captureStereoImagePair(self):
    self.log.pLogMsg(' [Open camera interfaces]')
    self.log.pLogMsg('')
    # Open Video capture for both cameras
    leftCamInterface  = cv.VideoCapture(self.LEFT_CAMERA_PORT)
    rightCamInterface = cv.VideoCapture(self.RIGHT_CAMERA_PORT)

    # Check if both webcams have been opened correctly
    if not leftCamInterface.isOpened():
      raise IOError("Cannot open webcam 01")
    if not rightCamInterface.isOpened():
      raise IOError("Cannot open webcam 02")

    # Initialize flags and counters
    suc1 = False
    suc2 = False
    attemptCount = 0 
    while (not suc1 and not suc2 or attemptCount < self.MAX_CONNECTION_ATTEMPTS ):
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
    elif showProduct == 'calibration':
      self.log.pLogMsg(f' >> Calibration Mode selected')
      self.log.pLogWrn(f'Captured images will be cropped and resized to {cnst.IMG_OUT_RESOLUTION_XY_PX} x {cnst.IMG_OUT_RESOLUTION_XY_PX} ')
      self.log.pLogMsg(f' ------------------------------ ')
      showMode   = self.SHOW_MODE_CALIBRATION
    elif showProduct == 'rectimg':
      showMode   = self.SHOW_MODE_RECTIFIED
    elif showProduct == 'dispmap':
      showMode   = self.SHOW_MODE_DISPARITY_MAP
    elif showProduct == 'laplace':
      showMode   = self.SHOW_MODE_LAPLCE_IMG
    else:
      self.log.pLogMsg('ERROR: showProduct not recognized. Using default show raw images.')
      showMode   = self.SHOW_MODE_RAW_IMG
    
    # Initialise image index for saved files 
    img_index  = 0

    self.log.pLogMsg(' [Open camera interfaces]')
    self.log.pLogMsg('')
    # Open Video capture for both cameras
    leftCamInterface  = cv.VideoCapture(self.LEFT_CAMERA_PORT)
    rightCamInterface = cv.VideoCapture(self.RIGHT_CAMERA_PORT)

    # Check if both webcams have been opened correctly
    if not leftCamInterface.isOpened():
      raise IOError("Cannot open webcam 01")
    if not rightCamInterface.isOpened():
      raise IOError("Cannot open webcam 02")
    
    self.doStreaming = True

    while self.doStreaming:
        if not isStartupMsgShown:
          isStartupMsgShown = True
          self.log.pLogMsg('')
          self.log.pLogMsg(' ==> [Start Streaming]')
          self.log.pLogMsg('')
          self._printKeyShortcuts()

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
            self.stereoImgGray = cv.cvtColor(self.imgToShow, cv.COLOR_BGR2GRAY) 
          elif showMode == self.SHOW_MODE_CALIBRATION:
            # Crop images to maximum resolution squared images
            self.imgl = self.imgl[:,cnst.IMG_OUT_CROP_X_PX:(cnst.C270_NATIVE_RESOLUTION_Y_PX + cnst.IMG_OUT_CROP_X_PX )]
            self.imgr = self.imgr[:,cnst.IMG_OUT_CROP_X_PX:(cnst.C270_NATIVE_RESOLUTION_Y_PX + cnst.IMG_OUT_CROP_X_PX )]
            # Resize image to output resolution
            self.imgl = cv.resize(self.imgl, (cnst.IMG_OUT_RESOLUTION_XY_PX, cnst.IMG_OUT_RESOLUTION_XY_PX))
            self.imgr = cv.resize(self.imgr, (cnst.IMG_OUT_RESOLUTION_XY_PX, cnst.IMG_OUT_RESOLUTION_XY_PX))
            # Concatenate stereo image
            self.imgToShow = cv.hconcat([self.imgl, self.imgr])
            # Assign output image 
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
          self._printExitStreamingMsg()
          break
        elif key == 27:
          # [ESC] -> exit
          self._printExitStreamingMsg()
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
    self._printExitStreamingMsg()

  def _printKeyShortcuts(self):
    self.log.pLogMsg(f'------------------------')
    self.log.pLogMsg(f' [Control Keys]')
    self.log.pLogMsg(f'------------------------')
    self.log.pLogMsg(f' [Esc] or [q] -> Quit streaming')
    self.log.pLogMsg(f' [c] -> Capture stereo image pair')
    self.log.pLogMsg(f' [s] -> Save capturing statistics')
    self.log.pLogMsg(f'')

  def _printExitStreamingMsg(self):
    self.log.pLogMsg(f'------------------------')
    self.log.pLogMsg(f' [Exit Streaming]')
    self.log.pLogMsg(f'------------------------')

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
        
        # Correct for any camera rotation if enabled
        if self.ENABLE_ROTATE_LEFT_IMG:
          self.imgl = cv.rotate(self.imgl, cv.ROTATE_180)
        if self.ENABLE_ROTATE_RIGHT_IMG:
          self.imgr = cv.rotate(self.imgr, cv.ROTATE_180)

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
  
  def exitStreaming(self):
    self.doStreaming = False
  
