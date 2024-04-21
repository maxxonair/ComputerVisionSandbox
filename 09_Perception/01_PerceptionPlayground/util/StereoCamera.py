#!/usr/bin/env python3
'''

@description: Class to interface home-build stereo camera using a USB interface 
              and OpenCV grab/retrieve functions to minimize the time difference 
              between the left and the right image. 
              
              Furthermore, this class allows to:
              - Save captured stereo frames to file 
              - monitor streaming metrics (time discrepency between left and right
                frame for every streamed frame)
              - Rectify the streamed images (given stereo calibration parameters
                are provided)
              - Apply filters to the streamed images (gaussian, laplacian)
              - Stream disparity map from stereo images
              
              This class was designed to be used for a home-build stereo camera
              made up of two inexpensive Logitech C270 (native resolution 
              1280x960) accomodated in a 3D printed stereo bench housing. The 
              plans for the 3D printed housing are accessible here:
              https://cad.onshape.com/documents/4b0896c0136d7c9a2105e3ca/w/84e41ab88d10607281bde922/e/79f058272cf29e7314daba29?renderMode=0&uiState=66093ffdbdbd147aae9ad8f1

              Parts required for this build: 
              * 2 x Logitech C270 HD Webcam
              * 1 x ISOUL USB Hub, 4-Port Ultra-Slim USB 2.0 Hub
              * 1 x Duttek USB C to USB adaptor
              * Optional: MPU650 + D1 mini for 3 axis gyro & accelerometer

@dependencies: This class requires OpenCV, as well as Matplotlib

Copyright © 2024 Max Braun

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the “Software”), to deal in the 
Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, subject 
to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
import cv2 as cv
import os
from time import perf_counter
import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from dataclasses import dataclass
from typing import Tuple

import util.image_functions as img
import util.constants as cnst

@dataclass
class StereoFace:
    """ Class for keeping track of key metrics from a stereo face detection """
    id: int
    center_coords_left_px: Tuple[int, int]
    center_coords_right_px: Tuple[int, int]
    radius_px: int
    
    def __init__(self, id, center_coords_left_px, center_coords_right_px, radius_px):
      self.id = id
      self.center_coords_left_px = center_coords_left_px
      self.center_coords_right_px = center_coords_right_px
      self.radius_px = radius_px

class StereoCamera:
  
  # Define imageShow window title:
  imgWindowName = "| > stereo bench stream [.mk1] < - | "

  # Streaming mode constants
  SHOW_MODE_NONE          = 0
  SHOW_MODE_DISPARITY_MAP = 1
  SHOW_MODE_RECTIFIED     = 2
  SHOW_MODE_RAW_IMG       = 3
  SHOW_MODE_LAPLCE_IMG    = 4
  # Calibration mode: Raw images are resized to calibration size 
  # Note: This mode must be used by streaming calls that rectify the image
  #       downstream!
  SHOW_MODE_CALIBRATION  = 5
  SHOW_MODE_FACE_RECOGNITION = 6

  MAX_CONNECTION_ATTEMPTS = 20

  # [!] Left and right stereo bench camera port ID
  # [!] Note: At the moment there is no reliable mechanism to determine which port
  #           is linked to the left and the right camera so these ports might be
  #           flipped. Start up camera first in streaming mode and ensure that 
  #           left and right images are at the correct location.
  LEFT_CAMERA_PORT  = 0
  RIGHT_CAMERA_PORT = 1

  # Correct for any camera rotation
  # If true -> Rotate image by 180 degree
  # This rotation is needed to compensate for the left camera beeing mounted 
  # upside down 
  # E.g. see this configuration 
  # https://cad.onshape.com/documents/4b0896c0136d7c9a2105e3ca/w/84e41ab88d10607281bde922/e/79f058272cf29e7314daba29?renderMode=0&uiState=66093b60bdbd147aae9ac972)
  ENABLE_ROTATE_LEFT_IMG  = True
  ENABLE_ROTATE_RIGHT_IMG = False
  
              
  # ---- Face recognition settings
  FACE_MARKER_THICKNESS = 1
  FACE_MARKER_COLOR     = (0,240,2)
  
  # Set confidence threshold to discard face recognition results
  FACE_RECOGN_CONF_THR  = 0.94
  
  # Default prefix when saving stereo frames to file
  IMG_PREFIX = 'stereoimg_'
  
  # Define HUD settings
  HUD_FONT_SCALE = 0.4
  HUD_TEXT_START_OFFSET_Y = 40
  HUD_TEXT_START_OFFSET_X = 20
  HUD_TEXT_STEP_Y = 20
  HUD_FONT = cv.FONT_HERSHEY_SIMPLEX
  #-----------------------------------------------------------------------------
  #                       [initialize]
  #-----------------------------------------------------------------------------
  def __init__(self, log):
    self.log = log
    self.imgl = []
    self.imgr = []
    
    self.stereoFacesList = []
    
    self.stereoImgGray = []
    # Initialize counter to count number of saved images
    self.saved_img_counter = 0
    # Initialize flag to track if any images have been saved
    self.is_img_saved = False
    # Initialize internal variable to track set class operational mode
    self.set_mode = self.SHOW_MODE_NONE
    # Initialize dictionary with meta data from this session. To be saved 
    # alongside the saved images
    self.session_meta_data = {
      "date": datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
      "img_prefix": self.IMG_PREFIX,
      "mode": self.set_mode,
      "number_of_images": self.saved_img_counter,
      "stereo_img_res": (0, 0),
      "single_img_res": (0, 0), 
      "y_cut_coord": 0, 
    }
  #-----------------------------------------------------------------------------
  #                       [public functions]
  #-----------------------------------------------------------------------------
  def captureStereoImagePair(self):
    self.log.pLogMsg(' [Open camera interfaces]')
    self.log.pLogMsg('')
    # Open Video capture for both cameras
    leftCamInterface  = cv.VideoCapture(self.LEFT_CAMERA_PORT)
    rightCamInterface = cv.VideoCapture(self.RIGHT_CAMERA_PORT)

    # Check if both webcams have been opened correctly
    if not leftCamInterface.isOpened():
      raise IOError("Cannot open camera #1")
    if not rightCamInterface.isOpened():
      raise IOError("Cannot open camera #2")

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

  def startStreaming(self, undist_maps, showProduct, showImages=False, camCalibration=None):
    # Setup folder to save images to 
    dateTimeStr = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    folder_name = f'{dateTimeStr}_frame_capture'
    # start time counter 
    self.path = os.path.join("output",folder_name)
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
      self.set_mode   = self.SHOW_MODE_RAW_IMG
    elif showProduct == 'calibration':
      self.log.pLogMsg(f' >> Calibration Mode selected')
      self.log.pLogWrn(f'Captured images will be cropped and resized to {cnst.IMG_OUT_RESOLUTION_XY_PX} x {cnst.IMG_OUT_RESOLUTION_XY_PX} ')
      self.log.pLogMsg(f' ------------------------------ ')
      self.set_mode   = self.SHOW_MODE_CALIBRATION
    elif showProduct == 'rectimg':
      self.set_mode   = self.SHOW_MODE_RECTIFIED
    elif showProduct == 'dispmap':
      self.set_mode   = self.SHOW_MODE_DISPARITY_MAP
    elif showProduct == 'laplace':
      self.set_mode   = self.SHOW_MODE_LAPLCE_IMG
    elif showProduct == 'facerecognition':
      self.set_mode   = self.SHOW_MODE_FACE_RECOGNITION
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      print('Running on device: {}'.format(device))
      self.mtcnn = MTCNN(keep_all=True, device=device)
    else:
      self.log.pLogMsg('ERROR: showProduct not recognized. Using default show raw images.')
      self.set_mode   = self.SHOW_MODE_RAW_IMG
    
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
        self.imgdim = self.imgl.shape
        grabDiffList.append(timetag_ms)

        if suc1 and suc2: 
          # Process and display image pair depending on selected mode
          if self.set_mode == self.SHOW_MODE_DISPARITY_MAP:
            # Rectify camera images 
            (rimgl, rimgr) = img.rectify_stereo_image(self.imgl, self.imgr, undist_maps)
            (rawDispMap, 
            dispMapFiltered, 
            dispMapFilteredAndNormalized) = img.create_disparity_map(rimgl, rimgr)
            self.imgToShow = dispMapFilteredAndNormalized
            # self.imgToShow = cv.applyColorMap(self.imgToShow, cv.COLORMAP_JET)
          elif self.set_mode == self.SHOW_MODE_RECTIFIED:\
            # Rectify camera images 
            (rimgl, rimgr) = img.rectify_stereo_image(self.imgl, self.imgr, undist_maps)
            self.imgToShow = cv.hconcat([rimgl, rimgr])
          elif self.set_mode == self.SHOW_MODE_RAW_IMG:
            self.imgToShow = cv.hconcat([self.imgl, self.imgr])
            self.stereoImgGray = cv.cvtColor(self.imgToShow, cv.COLOR_BGR2GRAY) 
          elif self.set_mode == self.SHOW_MODE_CALIBRATION:
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
          elif self.set_mode == self.SHOW_MODE_LAPLCE_IMG:
            kernel_size = 5
            # Rectify camera images 
            (rimgl, rimgr) = img.rectify_stereo_image(self.imgl, self.imgr, undist_maps)
            # Laplace transform rectified image pair 
            (rimgl, rimgr) = self._laplaceTransformStereoPair(rimgl, rimgr, kernel_size)
            # Show image
            self.imgToShow = cv.hconcat([rimgl, rimgr])
          elif self.set_mode == self.SHOW_MODE_FACE_RECOGNITION:
            # Crop images to maximum resolution squared images
            self.imgl = self.imgl[:,cnst.IMG_OUT_CROP_X_PX:(cnst.C270_NATIVE_RESOLUTION_Y_PX 
                                                            + cnst.IMG_OUT_CROP_X_PX )]
            self.imgr = self.imgr[:,cnst.IMG_OUT_CROP_X_PX:(cnst.C270_NATIVE_RESOLUTION_Y_PX 
                                                            + cnst.IMG_OUT_CROP_X_PX )]
            # Resize image to output resolution
            self.imgl = cv.resize(self.imgl, (cnst.IMG_OUT_RESOLUTION_XY_PX,
                                              cnst.IMG_OUT_RESOLUTION_XY_PX))
            self.imgr = cv.resize(self.imgr, (cnst.IMG_OUT_RESOLUTION_XY_PX,
                                              cnst.IMG_OUT_RESOLUTION_XY_PX))

            # Clear list of detected faces in stereo frame 
            self.stereoFacesList = []
            
            # Create copies to draw on
            self.imgDisplayL = self.imgl.copy()
            self.imgDisplayR = self.imgr.copy()
              
            # Detect faces in the left image
            boxesl, confl = self.mtcnn.detect(self.imgl)
            # Detect faces in the right image
            boxesr, confr = self.mtcnn.detect(self.imgr)
            
            # Initialize results metrics
            numFacesL = 0 
            numFacesR = 0
            faceCenterL = (0, 0)
            faceCenterR = (0, 0)
            
            if boxesl is not None:
              if confl[0] > self.FACE_RECOGN_CONF_THR:
                boxesl = np.asarray(boxesl, np.int32)
                boxDiml = boxesl.shape
                numFacesL = boxDiml[0]

                faceCenterL = (int((boxesl[0,0] + boxesl[0,2]) / 2),
                              int((boxesl[0,1] + boxesl[0,3]) / 2))
                self.imgDisplayL = cv.rectangle(self.imgDisplayL,
                                                (boxesl[0,0], boxesl[0,1]),
                                                (boxesl[0,2], boxesl[0,3]),
                                                self.FACE_MARKER_COLOR,
                                                self.FACE_MARKER_THICKNESS)
                self.imgDisplayL = cv.circle(self.imgDisplayL,faceCenterL,
                                             10, self.FACE_MARKER_COLOR, 1)
                self.imgDisplayL = cv.circle(self.imgDisplayL,faceCenterL,
                                             1, self.FACE_MARKER_COLOR, 1)
              
            if boxesr is not None:
              if confr[0] > self.FACE_RECOGN_CONF_THR:
                boxesr = np.asarray(boxesr, np.int32)
                boxDimr = boxesr.shape
                numFacesR = boxDimr[0]
                faceCenterR = (int((boxesr[0,0] + boxesr[0,2]) / 2),
                              int((boxesr[0,1] + boxesr[0,3]) / 2))
                self.imgDisplayR = cv.rectangle(self.imgDisplayR,
                                                (boxesr[0,0], boxesr[0,1]),
                                                (boxesr[0,2], boxesr[0,3]),
                                                self.FACE_MARKER_COLOR,
                                                self.FACE_MARKER_THICKNESS)
                
                self.imgDisplayR = cv.circle(self.imgDisplayR,faceCenterR,
                                             10, self.FACE_MARKER_COLOR,
                                             self.FACE_MARKER_THICKNESS)
                self.imgDisplayR = cv.circle(self.imgDisplayR,faceCenterR,
                                             1,
                                             self.FACE_MARKER_COLOR,
                                             self.FACE_MARKER_THICKNESS)
                
            # If a face is found in both images
            if camCalibration is not None:
              if numFacesL and numFacesR:
                
                # Compute number of faces seen in both frames
                # TODO these do not necessarily have to match. Something to be 
                # added to sort that out
                if numFacesL > numFacesR:
                  numFaces = numFacesR
                else:
                  numFaces = numFacesL
                  
                for faceId in range(numFaces):
                  self.stereoFacesList.append(StereoFace(faceId, faceCenterL, faceCenterR, 0))
                  
                p1 = np.zeros((3,1), np.float32)
                p2 = np.zeros((3,1), np.float32)
                v1 = img.getFeatureDirectionVector(faceCenterL,
                                                   np.asarray(camCalibration['K1']))
                
                v2 = img.getFeatureDirectionVector((faceCenterR[0],faceCenterL[1]),
                                                   np.asarray(camCalibration['K2']))
                
                # TODO: Stereo baseline currently not written to calibration files
                #       Add stereo baseline to calibration and remove this hard
                #       coded value
                # Note: The currently set value of 0.11807402364993869 m was 
                #       taken directly from the calibration log of the calibration
                #       linked at the time.
                p2[0] = 0.11807402364993869
                
                facePos_cam_m, faceDist_m = img.triangulateStereoFeature(p1=p1,
                                                              v1=v1,
                                                              p2=p2,
                                                              v2=v2)
                
                # Post-process position vector for displaying
                facePos_cam_m = np.asarray(facePos_cam_m)
                facePos_cam_m = np.squeeze(facePos_cam_m)

                self.imgDisplayL = cv.putText(self.imgDisplayL,
                                              f'Distance to person [m] : {faceDist_m:.2f}',
                                              (self.HUD_TEXT_START_OFFSET_X
                                              ,self.HUD_TEXT_START_OFFSET_Y),
                                              self.HUD_FONT,
                                              self.HUD_FONT_SCALE,
                                              self.FACE_MARKER_COLOR,
                                              1, cv.LINE_AA)
                self.imgDisplayL = cv.putText(self.imgDisplayL,
                                              f'Person position    [m] : {facePos_cam_m[0]:.2f} {facePos_cam_m[1]:.2f} {facePos_cam_m[2]:.2f}',
                                              (self.HUD_TEXT_START_OFFSET_X
                                              ,self.HUD_TEXT_START_OFFSET_Y+self.HUD_TEXT_STEP_Y),
                                              self.HUD_FONT,
                                              self.HUD_FONT_SCALE,
                                              self.FACE_MARKER_COLOR,
                                              1, cv.LINE_AA)
            else: 
              numFaces = 10
              self.log.pLogWrn(f'No camera calibration data given. Face localisation disabled.')
              self.imgDisplayL = cv.putText(self.imgDisplayL,
                                            f'Distance to person [m] : N/A',
                                            (self.HUD_TEXT_START_OFFSET_X
                                            ,self.HUD_TEXT_START_OFFSET_Y),
                                            self.HUD_FONT,
                                            self.HUD_FONT_SCALE,
                                            self.FACE_MARKER_COLOR,
                                            1, cv.LINE_AA)
              self.imgDisplayL = cv.putText(self.imgDisplayL,
                                            f'Person position    [m] : N/A',
                                            (self.HUD_TEXT_START_OFFSET_X
                                            ,self.HUD_TEXT_START_OFFSET_Y+self.HUD_TEXT_STEP_Y),
                                            self.HUD_FONT,
                                            self.HUD_FONT_SCALE,
                                            self.FACE_MARKER_COLOR,
                                            1, cv.LINE_AA)
            
            # self.log.pLogMsg(f'Number of faces detected by both eyes: {numFaces} ')
              
            # Compile output image 
            self.imgToShow = cv.hconcat([self.imgDisplayL, self.imgDisplayR])
          else:
            self.log.pLogMsg("[ERR] Error self.set_mode not valid. Exiting")
            break

          # Show image pair 
          if showImages:
            cv.imshow(f'{self.imgWindowName} | image resolution: {cnst.IMG_OUT_RESOLUTION_XY_PX} | Q/ESC - quit | C - capture image | S - save statistics', self.imgToShow)
        else:
          self.log.pLogMsg("Image retrieval failed.")
            
        # Wait for key inputs
        key = cv.waitKey(1)
        # Define key input actions
        if key == ord('q'):
          # [q] -> exit
          self._callback_on_exit_fnct()
          break
        elif key == 'esc':
          # [ESC] -> exit
          self._callback_on_exit_fnct()
          break
        elif key == 27:
          # [ESC] -> exit
          self._callback_on_exit_fnct()
          break
        elif key == ord('c'):
          # [c] -> Save image to file
          if not isFolderSetup:
            # Create time tagged folder to save image pairs to
            os.mkdir(self.path)
            isFolderSetup = True
          # Define image name 
          img_name = f'{self.IMG_PREFIX}{img_index}.png'
          # Define complete image save path 
          img_file_path = os.path.join(self.path,img_name)
          self.log.pLogMsg(f'Save image to file: {img_file_path}')
          self.log.pLogMsg(f'Image dimensions: {(self.imgToShow).shape}')
          cv.imwrite(img_file_path, self.imgToShow) 
          img_index += 1 
          self.saved_img_counter += 1
          self.is_img_saved = True
        elif key == ord('s'):
          # [s] -> Save statistics
          if not isFolderSetup:
            # Create time tagged folder to save image pairs to
            os.mkdir(self.path)
            isFolderSetup = True
          # Define image name 
          csv_file_name = "grab_timing_statistics.csv"
          # Define complete image save path 
          csv_full_file_path = os.path.join(self.path,csv_file_name)
          png_full_file_path = os.path.join(self.path,'grab_timing_statistics.png')
          self.log.pLogMsg("Save statistics to file: "+csv_full_file_path)
          # Save complete grabDiffList as csv
          np.savetxt(csv_full_file_path, grabDiffList, delimiter=",")
          # Save histogram as png
          self._createGrabTimingHistogram(grabDiffList, png_full_file_path)

    leftCamInterface.release()
    rightCamInterface.release()
    cv.destroyAllWindows()
    self._callback_on_exit_fnct()
  
  def exitStreaming(self):
    '''
    Public callback function to deassert flag and end streaming
    '''
    self.doStreaming = False
  #-----------------------------------------------------------------------------
  #                       [private functions]
  #-----------------------------------------------------------------------------
  def _printKeyShortcuts(self):
    self.log.pLogMsg(f'------------------------')
    self.log.pLogMsg(f' [Control Keys]')
    self.log.pLogMsg(f'------------------------')
    self.log.pLogMsg(f' [Esc] or [q] -> Quit streaming')
    self.log.pLogMsg(f' [c]          -> Capture stereo image pair')
    self.log.pLogMsg(f' [s]          -> Save capturing statistics')
    self.log.pLogMsg(f'')

  def _callback_on_exit_fnct(self):
    '''
    This callback shall contain all functionality to be called when exiting 
    image streaming.
    
    '''
    self.log.pLogMsg(f'------------------------')
    self.log.pLogMsg(f'   >[Exit Streaming]<   ')
    self.log.pLogMsg(f'------------------------')
    # Check if images have been saved this session
    # if so -> Save meta data file alongside
    if self.is_img_saved:
      # Update image capture meta data dictionary
      self.session_meta_data['mode'] = self.set_mode
      self.session_meta_data['number_of_images'] = self.saved_img_counter
      # TODO: This is a shortcut and only valid if running in 'calibration' mode
      #       To be fixed and implemented properly
      self.session_meta_data['stereo_img_res'] = (cnst.IMG_OUT_RESOLUTION_XY_PX, 2*cnst.IMG_OUT_RESOLUTION_XY_PX)
      self.session_meta_data['single_img_res'] = (cnst.IMG_OUT_RESOLUTION_XY_PX, cnst.IMG_OUT_RESOLUTION_XY_PX)
      self.session_meta_data['y_cut_coord']    = cnst.IMG_OUT_RESOLUTION_XY_PX
      
      # Serializing dictionary to json object
      json_object = json.dumps(self.session_meta_data, indent=4)
      
      # Save image capture meta data file to json
      temp_save_path = os.path.join(self.path, "stereo_img_meta.json")
      self.log.pLogMsg(f'Saving meta data to: {temp_save_path}')
      with open(temp_save_path, "w") as outfile:
        outfile.write(json_object)
      
  def _acquireStereoImagePair(self, leftCamInterface, rightCamInterface, isPromptGrabDelay):
    '''
    Function to acquire stereo image pair from the cameras using grab() and 
    retrieve() methods
    
    
    @returns:
    suc1 - flag true if left image acquisition was successful
    imgl - left image 
    suc2 - flag true if right image acquisition was successful
    imgr - right image
    timetag_ms - Time between left and right image in milli seconds
    
    '''
    # Grab frames from both cameras
    t_start = perf_counter()
    ret1 = leftCamInterface.grab()
    t_end = perf_counter()
    ret2 = rightCamInterface.grab()
    timetag_ms = ( t_end - t_start ) * 1000

    # Initialize success flags for left and right images
    suc1 = []
    suc2 = []
    # Initialize left and right images as empty arrays
    self.imgl = []
    self.imgr = []

    # Print time taken to grab one cameras image
    if isPromptGrabDelay:
      self.log.pLogMsg("Delay between left and right frame [ms] : "+str(timetag_ms))
    if not ret1 and not ret2:
      self.log.pLogErr("Failed to grab both frames")  
      if not ret1 :
        self.log.pLogErr("Failed to grab camera 1 frame")
      if not ret2 :
        self.log.pLogErr("Failed to grab camera 2 frame")
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

    return (suc1,self.imgl, suc2, self.imgr, timetag_ms)

  def _createGrabTimingHistogram(self, grabDiffList, pngFilePath):
    '''
    Create histogram plot of recorded time differences between left and right
    images for all stereo frames.
    '''
    plt.hist(grabDiffList, bins=60, range=[0, 60], color = "gray")
    plt.title('Stereo camera session: Grab frame difference')
    plt.xlabel('Time between left and right frame [ms]')
    plt.savefig(pngFilePath)

  def _laplaceTransformStereoPair(self, limg, rimg, kernel_size):
    '''
    Apply laplace filter to left and right image of stereo frame
    '''
    ddepth = cv.CV_8U
    limg = cv.Laplacian(limg, ddepth, ksize=kernel_size)
    rimg = cv.Laplacian(rimg, ddepth, ksize=kernel_size)
    return limg, rimg
  
