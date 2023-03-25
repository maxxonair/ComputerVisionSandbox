
"""
Test class to introduce live charts showing processed streaming content 
from the stereo camera 


"""

import matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

import numpy as np
import cv2 as cv

from util.stereo_camera import StereoCamera
import util.image_functions as img

class MonitorChart:

  worldPoints_m_Cam  = []
  
  def __init__(self, boardDimensions, R_mat, T_mat, undistMaps, log):
    
    # Compose projection matrices 
    self.P1 = np.hstack((np.eye(3, dtype=float), np.zeros((3,1), dtype=float)))
    self.P2 = np.hstack((R_mat, T_mat))
    self.boardSize  = boardDimensions
    self.undistMaps = undistMaps
    self.log        = log
    
    self.camera = StereoCamera(log)
    
    self.isStreamingStarted = False

    self.fig  = plt.figure(figsize=(20, 10))
    self.ax1  = self.fig.add_subplot(1,2,1)
    self.ax3D = self.fig.add_subplot(1,2,2, projection='3d')
    
  def _findImgPoints(self, rimg):
    imgPoints   = []
    successFlag = False
    
    readFlags  = cv.CALIB_CB_ADAPTIVE_THRESH
    readFlags |= cv.CALIB_CB_FILTER_QUADS
    readFlags |= cv.CALIB_CB_NORMALIZE_IMAGE
    
    subPixCriteria = (cv.TERM_CRITERIA_EPS +
                      cv.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    # Find the chess board corners
    (patternFound, 
     cornerPoints) = cv.findChessboardCorners(rimg,
                                                self.boardSize,
                                                flags=readFlags)
    if patternFound == True:
      successFlag = True
      # If Chessboard calibration -> run cornerSubPix refinement
      imgPoints = cv.cornerSubPix(rimg,
                                  cornerPoints,
                                  (11, 11),
                                  (-1, -1),
                                  subPixCriteria)
    return (successFlag, imgPoints)
    
  def _drawChessboardOnImage(self, rimg, imgPoints):
    img_out =  cv.cvtColor(rimg, cv.COLOR_GRAY2BGR)
    cv.drawChessboardCorners(img_out,
                            self.boardSize,
                            imgPoints,
                            True)
    return img_out
    
  def _compPatternWorldPoints(self, rimgl, rimgr):
    (w,h) = rimgl.shape
    
    self.rimg = cv.hconcat([rimgl, rimgr])
    
    self.suc1, self.imgPointsl = self._findImgPoints(rimgl)
    self.suc2, self.imgPointsr = self._findImgPoints(rimgr)
    
    if self.suc1 and self.suc2:
      #  Normalize image points
      self.normImgPointsl = self.imgPointsl / w
      self.normImgPointsr = self.imgPointsr / w
      
      # Triangulate stereo image points to world points
      homogeneusWorldPoints = cv.triangulatePoints(self.P1, 
                                                    self.P2, 
                                                    self.normImgPointsl, 
                                                    self.normImgPointsr)
      
      # Transform homogenous to inhomogenous coordinates
      (w, numCorners) = homogeneusWorldPoints.shape
      
      # Check that output is in the format that is exptexted
      if w != 4:
        self.log.pLogErr('Homogeneous corrdinates are in wrong format. Exiting')
        exit(1)
        
      self.worldPoints_m_Cam = np.zeros((numCorners,3), dtype=float)
      for iCounter in range(numCorners):
        self.worldPoints_m_Cam[iCounter,:] = (homogeneusWorldPoints[:3, iCounter] / homogeneusWorldPoints[3, iCounter]) 
    
  def _createChart(self):

    self.ax3D.set_aspect('equal')
    
    #----------------------------------------------------------
    #           [Data charts]
    #----------------------------------------------------------
    self.ax1.clear()
    self.ax3D.clear()
    # If pattern data is available -> Draw it on the displayed image
    if self.suc1 and self.suc2:
      cimgl = self._drawChessboardOnImage(self.gimgl, self.imgPointsl)
      cimgr = self._drawChessboardOnImage(self.gimgr, self.imgPointsr)
      imgToShow = cv.hconcat([cimgl, cimgr])
      self.ax1.imshow(imgToShow, cmap='gray')
    else:
      # If not -> only show the image itself
      self.ax1.imshow(self.rimg, cmap='gray')
      
    if len(self.worldPoints_m_Cam) != 0:
      try:
        self.ax3D.scatter(self.worldPoints_m_Cam[:,0],
                          self.worldPoints_m_Cam[:,1],
                          self.worldPoints_m_Cam[:,2])
      except:
        print('Plotting world points failed')
    else:
      print('No pattern detected in image frame')
    #----------------------------------------------------------
    minValCameraFrame = -0.5
    maxValCameraFrame =  2
    self.ax1.set_title('Monitor test')
    self.ax3D.set_xlabel('x / m')
    self.ax3D.set_ylabel('y / m')
    self.ax3D.set_zlabel('z / m')
    self.ax3D.axes.set_xlim3d(left=minValCameraFrame, right=maxValCameraFrame)
    self.ax3D.axes.set_ylim3d(bottom=minValCameraFrame, top=maxValCameraFrame) 
    self.ax3D.axes.set_zlim3d(bottom=minValCameraFrame, top=maxValCameraFrame) 
    self.fig.tight_layout()
    
  def run(self):
    # self.update for static image tests
    self._createChart()
    
  def update(self, iFrame):
    # [Capture stereo image from camera]
    if not self.isStreamingStarted:
      streamingMode = 'rawimg'
      self.streamingThread = threading.Thread(target=self.camera.startStreaming, name="Stream", args=(self.undistMaps, streamingMode, False))
      self.streamingThread.start()
      self.isStreamingStarted = True

    # Convert color images to grayscale
    if len(self.camera.stereoImgGray) != 0 :
      (h, w) = self.camera.stereoImgGray.shape[:2]
      # Split dual image into left&right image
      # + -----> w/x
      # | [][][][][][][][][][][][][][][]
      # | [][][][][][][][][][][][][][][]
      # | [][][][][][][][][][][][][][][]
      # | [][][][][][][][][][][][][][][]
      # V
      # h/y             | w2
      #    left image  < > right image
      #
      # Compute separator x position
      w2 = int(w/2)
      # Extract left & right image
      self.gimgl  = self.camera.stereoImgGray[ 0:h , 0:w2 ]
      self.gimgr  = self.camera.stereoImgGray[ 0:h , w2:w ]

      self.imageSize = self.gimgl.shape[:2]

      # [Rectify raw images]
      (rimgl, 
       rimgr) = img.rectifyStereoImageSet(self.gimgl, 
                                          self.gimgr, 
                                          self.undistMaps)
      
      print(rimgl.shape)
      self._compPatternWorldPoints(rimgl, rimgr)
      self._createChart()

  def monitor(self, refreshInterval_s):
    self.MonitorAnimation = animation.FuncAnimation(self.fig, 
                                             self.update, 
                                             interval=refreshInterval_s*1000)
    plt.show()