
"""
Test class to introduce live charts showing processed streaming content 
from the stereo camera 


"""

import matplotlib 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import threading

import signal

import numpy as np
import cv2 as cv

from util.StereoCamera import StereoCamera
import util.image_functions as img
from util.ChartClasses.CoordinateSystem import Frame
import util.constants as cnst

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

class MonitorChart:

  worldPoints_m_Cam  = []
  
  def __init__(self, boardDimensions, CameraCalibration, undistMaps, log):
    
    # Compose projection matricesƒ ƒ
    self.CameraCalibration = CameraCalibration
    self.boardSize  = boardDimensions
    self.undistMaps = undistMaps
    self.log        = log
    
    self.camera = StereoCamera(log)
    
    self.isStreamingStarted = False
    
    self.camera_frame = Frame()

    self.isSetAxesOrientation = False

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
    self.worldPoints_m_Cam = []
    
    self.rimg = cv.hconcat([rimgl, rimgr])
    
    self.suc1, self.imgPointsl = self._findImgPoints(rimgl)
    self.suc2, self.imgPointsr = self._findImgPoints(rimgr)
    
    if self.suc1 and self.suc2:
      #  Normalize image points
      self.normImgPointsl = self.imgPointsl / w
      self.normImgPointsr = self.imgPointsr / w

      self.P1 = np.hstack((np.eye(3, dtype=float), np.zeros((3,1), dtype=float)))
      # Decompose calibration projection matrices 
      K1,R1,homT1, _, _, _, _ = cv.decomposeProjectionMatrix(self.CameraCalibration['P1'])
      K2,R2,homT2, _, _, _, _ = cv.decomposeProjectionMatrix(self.CameraCalibration['P2'])

      # ---------------------------------------
      # TODO: Remove this. Temporarily here for debugging and exploring.
      # fx = K1[0,0]
      # fy = K1[1,1]

      # Fx_m = fx * cnst.C270_SENSOR_WIDTH_M / 600
      # Fy_m = fy * cnst.C270_SENSOR_HEIGHT_M / 600

      # self.log.pLogMsg(f'Fx [m]: {Fx_m}')
      # self.log.pLogMsg(f'Fy [m]: {Fy_m}')
      # ---------------------------------------


      # Transform translation vector from homogeneous to inhomogeneous coordinates
      T1 = (homT1[:3] / homT1[-1])[:3]
      T2 = (homT2[:3] / homT2[-1])[:3]

      # Normalize intrinsic matrices 
      row_sums = K1.sum(axis=1)
      normK1 = K1 / row_sums[:, np.newaxis]
      row_sums = K2.sum(axis=1)
      normK2 = K2 / row_sums[:, np.newaxis]

      # Build [R|T] for left and right camera
      RT1 = np.hstack((R1, T1), dtype=float)
      RT2 = np.hstack((R2, T2), dtype=float)

      # Compose semi-normalised projection matrices for left and right camera
      P1 = np.dot(normK1, RT1)
      P2 = np.dot(normK2, RT2)
      
      # Triangulate stereo image points to world points
      homogeneusWorldPoints = cv.triangulatePoints(P1, 
                                                   P2, 
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
        self.worldPoints_m_Cam[iCounter,:] = (homogeneusWorldPoints[:3, iCounter] / - homogeneusWorldPoints[3, iCounter]) 

  def _createChart(self):

    self.ax3D.set_aspect('equal')
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    #----------------------------------------------------------
    #           [Data charts]
    #==========================================================
    try:
      self.ax1.clear()
    except:
      DoNothing = True
    self.ax3D.clear()
    #-------------------------------------------

    # If pattern data is available -> Draw it on the displayed image
    if self.suc1 and self.suc2:
      cimgl = self._drawChessboardOnImage(self.rimgl, self.imgPointsl)
      cimgr = self._drawChessboardOnImage(self.rimgr, self.imgPointsr)
      imgToShow = cv.hconcat([cimgl, cimgr])
      self.ax1.imshow(imgToShow, cmap='gray')
    else:
      # If not -> only show the image itself
      self.ax1.imshow(self.rimg, cmap='gray')
    #-------------------------------------------

    if len(self.worldPoints_m_Cam) != 0:
      try:
        self.ax3D.scatter(self.worldPoints_m_Cam[:,0],
                          self.worldPoints_m_Cam[:,1],
                          self.worldPoints_m_Cam[:,2])
      except:
        print('Plotting world points failed')
    else:
      print('No pattern detected in image frame')
    #==========================================================
    minValCameraFrame = -1.2
    maxValCameraFrame =  2

    # Set callback on close event 
    self.fig.canvas.mpl_connect('close_event', self._callbackOnWindowClose)

    # Set chart titles
    self.ax1.set_title('[stereo camera image]')
    self.ax3D.set_title('[corner points in left camera frame]')

    # Set 3D chart axis names
    self.ax3D.set_xlabel('x / m')
    self.ax3D.set_ylabel('y / m')
    self.ax3D.set_zlabel('z / m')
    
    # self.ax1 = self.camera_frame.draw(self.ax1)

    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

    a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='r')
    self.ax3D.add_artist(a)
    a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='b')
    self.ax3D.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='g')
    self.ax3D.add_artist(a)

    # Give them a name:
    self.ax3D.text(0.0, 0.0, -0.1, r'$0$')
    self.ax3D.text(1.1, 0, 0, r'$x$')
    self.ax3D.text(0, 1.1, 0, r'$y$')
    self.ax3D.text(0, 0, 1.1, r'$z$')

    if self.isSetAxesOrientation:
      self.isSetAxesOrientation = True
      self.ax3D.view_init(elev=-40.0, azim=-90.0)

    # Set 3D chart axis limits
    self.ax3D.axes.set_xlim3d(left=minValCameraFrame, right=maxValCameraFrame)
    self.ax3D.axes.set_ylim3d(bottom=minValCameraFrame, top=maxValCameraFrame) 
    self.ax3D.axes.set_zlim3d(bottom=minValCameraFrame, top=maxValCameraFrame) 
    self.fig.tight_layout()
    self.fig.canvas.draw()
    self.fig.show()
    
  def run(self):
    # self.update for static image tests
    self._createChart()
    
  def update(self, iFrame):
    # [Capture stereo image from camera]
    if not self.isStreamingStarted:
      streamingMode = 'calibration'
      # Start stereo camera stream in separate thread
      self.streamingThread = threading.Thread(target=self.camera.startStreaming, 
                                              name="Stream", 
                                              args=(self.undistMaps, streamingMode, False))
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
      (self.rimgl, 
       self.rimgr) = img.rectify_stereo_image(self.gimgl, 
                                          self.gimgr, 
                                          self.undistMaps)
      
      self._compPatternWorldPoints(self.rimgl, self.rimgr)
      self._createChart()

  def monitor(self, refreshInterval_s):
    self.MonitorAnimation = animation.FuncAnimation(self.fig, 
                                                    self.update, 
                                                    interval=refreshInterval_s*1000)
    plt.show()

  def _callbackOnWindowClose(self, event):
    self.camera.exitStreaming()
    self.streamingThread.join()
