#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:43:57 2022

@author: x
"""
from tkinter import W
import cv2 as cv
from tqdm import tqdm
import glob
import numpy as np

from util.CameraMetaData import CameraData
from util.CalibImageData import CalibImageData
from util.StereoCalibrationData import StereoCalibrationData


class StereoCalibrator:
    # --------------------------------------------------------------------------
    # Set global variables
    # --------------------------------------------------------------------------
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    leftImgpoints  = []
    rightImgpoints = []

    # Internal state flags
    # Flag calibration run completed
    bFlagCalibrationComplete = False

    # Settings flags
    bFlagEnableImgScaling = False
    bFlagSaveMarkedImages = False
    bFlagSaveScaledImages = False
    bFlagUseRemapping     = False
    enableUseBlobDetector = True

    # File Paths
    sInputFilePath      = ''
    sProcessedImagePath = ''
    sScaledImgPath      = ''
    sParameterFilePath  = ''
    sRecifiedImgPath    = ''

    # Process images at scale:
    scale_percent = 100

    # Chessboard pattern dimension
    # ( rows, columns)
    boardSize = (0, 0)

    # Number of images in input set
    iNrImagesInSet = 0

    aSuccIndexList = []

    # Camera claibration paramters
    ret = 0
    aKmat = []
    aDist = []

    totalReprojectionError = 0

    rectificationAlpha = 0

    imageSize = []
    # --------------------------------------------------------------------------
    #       OpenCV calibration setting parameters 
    # --------------------------------------------------------------------------
    subPixCriteria = (cv.TERM_CRITERIA_EPS +
                      cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # Initialize stereo calibration flags
    stereoCalibFlags = 0
    # Set flag settings
    stereoCalibFlags |= cv.CALIB_FIX_FOCAL_LENGTH
    stereoCalibFlags |= cv.CALIB_FIX_INTRINSIC
    stereoCalibFlags |= cv.CALIB_USE_INTRINSIC_GUESS
    stereoCalibFlags |= cv.CALIB_ZERO_TANGENT_DIST

    stereoCalibCriteria = (cv.TERM_CRITERIA_MAX_ITER +
                        cv.TERM_CRITERIA_EPS, 250, 1e-6)
    
    rectificationFlags = cv.CALIB_ZERO_DISPARITY

    # alpha=-1 -> Let OpenCV optimize black parts.

    # alpha= 0 -> Rotate and cut the image so that there will be no black
    # parts. This option cuts the image so badly most of the time, that
    # you won’t have a decent high-quality image but worth to try.

    # alpha= 1 -> Make the transform but don’t cut anything.
    stereoCalibAlpha = -1
    # --------------------------------------------------------------------------
    # Restrict the number of calibration images to a maximum value:
    maxNrCalibrationImages = -1
    
    # Image list 
    # track reprojection errors and include flags
    aImageList = []

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # --------------------------------------------------------------------------
    #   >> Setters and Getters
    # --------------------------------------------------------------------------

    def setEnableRemapping(self, bFlagUseRemapping):
        self.bFlagUseRemapping = bFlagUseRemapping

    def setEnableImgScaling(self, bFlagEnableImgScaling):
        self.bFlagEnableImgScaling = bFlagEnableImgScaling

    def setEnableMarkedImages(self, bFlagSaveMarkedImages):
        self.bFlagSaveMarkedImages = bFlagSaveMarkedImages

    def setEnableSaveScaledImages(self, bFlagSaveScaledImages):
        self.bFlagSaveScaledImages = bFlagSaveScaledImages
        
    def setEnableUseBlobDetector(self, bFlagUseBlobDetector):
        self.enableUseBlobDetector = bFlagUseBlobDetector
        
    def setCalibrationPatternType(self,  sPatternType):
        self.sPatternType = sPatternType

    def getObjectPoints(self):
        return self.objpoints
    
    def setCropRectifiedImages(self, bFlagCropRectifImages):
        self.bFlagCropRectifImages = bFlagCropRectifImages
    # --------------------------------------------------------------------------
    #   >> Init
    # --------------------------------------------------------------------------

    def __init__(self, sInputFilePath, sProcessedImagePath, sScaledImgPath,
                 sParameterFilePath, sRecifiedImgPath, sDisparityMapsPath,
                 boardSize, objectSize, sPatternType, log):
        self.log = log
        self.sInputFilePath         = sInputFilePath
        self.sProcessedImagePath    = sProcessedImagePath
        self.sScaledImgPath         = sScaledImgPath
        self.sParameterFilePath     = sParameterFilePath
        self.sRecifiedImgPath       = sRecifiedImgPath
        self.sDisparityMapsPath     = sDisparityMapsPath
        self.sPatternType           = sPatternType

        self.boardSize              = boardSize
        
        self.objectSize             = objectSize

        self.aStereoImagePairs      = []

        self.LeftCamera  = CameraData("Left_Camera")
        self.RightCamera = CameraData("Right_Camera")
        
        self.stereoCalibData = StereoCalibrationData(self.log)

        # Prepare object points
        if self.sPatternType == "acircles":
            self._createAcircleObjectPoints()
        elif self.sPatternType == "chessboard":
            self._createChessboardObjectPoints()
        else:
            self.pLogErr("Calibration board pattern type not valid!")
    
    # --------------------------------------------------------------------------
    #   >> Class functions Stereo Calibration
    # --------------------------------------------------------------------------
    def readStereoPairs(self):
        # List calibration images:
        self.dual_images    = glob.glob(self.sInputFilePath+"*")
        imageIndex          = 0
        self.iNrImagesInSet = 0
        # Reset success index list
        self.aSuccIndexList = []
        
        self.aImageList     = []

        for ii, fname in tqdm( enumerate( self.dual_images ) ):
            # Load dual image
            img = cv.imread(fname)
            # Convert image to greyscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            outputImage = gray
            outputColor = img
            
            (h, w) = gray.shape[:2]
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
            leftImage  = gray[ 0:h , 0:w2 ]
            rightImage = gray[ 0:h , w2:w ]
            
            leftImageColor  = img[ 0:h , 0:w2 ]
            rightImageColor = img[ 0:h , w2:w ]
            
            self.imageSize = leftImage.shape[:2]
            
            # Make sure left and right images are the same size
            if np.all( leftImage.shape[:2] == rightImage.shape[:2]) != True:
                self.log.pLogErr("Left and Right images are not the same size. Exiting!")
                return

            # Print status
            self.log.pLogMsg('Load image pair : ' + str(imageIndex))
            patternFound = False

            if self.sPatternType is  "acircles":
                # Set flags for assymetric circle pattern detection
                readFlags = cv.CALIB_CB_CLUSTERING
                readFlags |= cv.CALIB_CB_ASYMMETRIC_GRID
                
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                
                # Create blob detector to support finding assymetric 
                if self.enableUseBlobDetector:
                    self._createBlobDetector()
                else:
                    self.blobDetector = None
                
                patternFound_left, corners_left = cv.findCirclesGrid( leftImage, 
                                                                    self.boardSize, 
                                                                    blobDetector=self.blobDetector,
                                                                    flags=readFlags)
                
                patternFound_right, corners_right = cv.findCirclesGrid( rightImage, 
                                                                    self.boardSize, 
                                                                    blobDetector=self.blobDetector,  
                                                                    flags=readFlags)
            elif self.sPatternType is  "chessboard":
                readFlags = cv.CALIB_CB_ADAPTIVE_THRESH
                readFlags |= cv.CALIB_CB_FILTER_QUADS
                readFlags |= cv.CALIB_CB_NORMALIZE_IMAGE
                # Find the chess board corners
                patternFound_left, corners_left = cv.findChessboardCorners( leftImage,
                                                                            self.boardSize,
                                                                            flags=readFlags)
                patternFound_right, corners_right = cv.findChessboardCorners( rightImage,
                                                                            self.boardSize,
                                                                            flags=readFlags)

            # Create imageData instance 
            calibImageData = CalibImageData()
            
            # If found, add object points, image points (after refining them)
            if patternFound_left == True and patternFound_right == True:
                patternFound = True
                # If Chessboard calibration -> run cornerSubPix refinement
                if self.sPatternType is  "chessboard":
                    corners2_left = cv.cornerSubPix(leftImage, 
                                                    corners_left,
                                                    (11, 11),
                                                    (-1, -1),
                                                    self.subPixCriteria)
                    
                if self.sPatternType is  "chessboard":
                    corners2_right = cv.cornerSubPix(rightImage, 
                                                     corners_right,
                                                    (11, 11),
                                                    (-1, -1),
                                                    self.subPixCriteria)
                
                # Update success state
                self.bFlagCalibrationComplete = True
                self.aSuccIndexList.append(imageIndex)

                self.iNrImagesInSet = self.iNrImagesInSet + 1
                # Save scaled images with drawn corners
                if self.bFlagSaveMarkedImages:
                    cv.drawChessboardCorners(leftImageColor,
                                             self.boardSize,
                                             corners_left,
                                             patternFound)
                    cv.drawChessboardCorners(rightImageColor,
                                             self.boardSize,
                                             corners_right,
                                             patternFound)
                    fileName = str(imageIndex) + '_calib.png'
                    savePath = self.sProcessedImagePath + fileName
                    self.log.pLogMsg("Result images saved to: "+savePath)
                    dual_image = cv.hconcat([leftImageColor, rightImageColor])
                    cv.imwrite(savePath, dual_image)
                if self.bFlagSaveScaledImages:
                    TBD = True
            else:
                if patternFound_left == True and patternFound_right == False:
                    self.log.pLogMsg(str(imageIndex)+")  [No calibration pattern found in right image].")
                elif patternFound_left == False and patternFound_right == True:
                    self.log.pLogMsg(str(imageIndex)+")  [No calibration pattern found in left image].")
                else :
                    self.log.pLogMsg(str(imageIndex)+")  [No calibration pattern found in both images].")
            
            # Set Values:
            calibImageData.imageFilePath        = fname
            calibImageData.imageIndex           = ii
            calibImageData.isPatternFound       = patternFound
            calibImageData.leftImagePoints      = corners_left
            calibImageData.rightImagePoints     = corners_right
            # Append image data to list
            self.aImageList.append(calibImageData)
            
            imageIndex = imageIndex + 1


        self.log.pLogMsg('>> Pattern found in '
                         + str(self.iNrImagesInSet)
                         + '/'+str(imageIndex)
                         + " images. %.2f percent yield" %
                         (self.iNrImagesInSet/imageIndex * 100))


        self._monoCalibrate()
    
    # Function: Run stereo calibration and save images 
    def stereoCalibrate(self):
        self.log.pLogMsg("")
        self.log.pLogMsg("[RUN STEREO CALIBRATION]")
        self.log.pLogMsg("")

        # Check if any image pairs are in image buffer
        if len(self.aImageListToObjPointList) == 0:
            self.log.pLogErr(
                "List of valid stereo image pairs (pattern found) is empty. Exiting.")
            return False

        (flagStereoCalibrationSucceeded, 
         self.stereoCalibData.K1, 
         self.stereoCalibData.D1, 
         self.stereoCalibData.K2, 
         self.stereoCalibData.D2, 
         self.stereoCalibData.R, 
         self.stereoCalibData.T, 
         self.stereoCalibData.E, 
         self.stereoCalibData.F
         ) = cv.stereoCalibrate(
         self.objpoints,
         self.leftImgpoints,
         self.rightImgpoints,
         self.LeftCamera.Kmat,
         self.LeftCamera.Dvec,
         self.RightCamera.Kmat,
         self.RightCamera.Dvec,
         self.imageSize,
         criteria=self.stereoCalibCriteria,
         flags=self.stereoCalibFlags
         )
        
        if flagStereoCalibrationSucceeded:
            (self.stereoCalibData.R1,
            self.stereoCalibData.R2,
            self.stereoCalibData.P1,
            self.stereoCalibData.P2,
            self.stereoCalibData.Q,
            roi_left,
            roi_right
            ) = cv.stereoRectify(
            cameraMatrix1=self.stereoCalibData.K1,
            cameraMatrix2=self.stereoCalibData.K2,
            distCoeffs1=self.stereoCalibData.D1,
            distCoeffs2=self.stereoCalibData.D2,
            imageSize=self.imageSize,
            R=self.stereoCalibData.R,
            T=self.stereoCalibData.T,
            flags=self.rectificationFlags,
            alpha=self.stereoCalibAlpha,
            newImageSize=self.imageSize
            )

            self.log.pLogMsg("")
            self.log.pLogMsg("Stereo Rectification completed")
            self.log.pLogMsg("")
            
            # Print complete set of stereo calibration parameters
            self.stereoCalibData.printStereoCalibrationData()

            # Update calibration status for both cameras
            self.LeftCamera.setStereoCalibrated()
            self.RightCamera.setStereoCalibrated()
            
            # Calculate reprojection erro statistics 
            self._showReprojectionError()
            
            # Save calibration results to file 
            self._saveResults()
            
        else:
            self.log.pLogErr("")
            self.log.pLogErr("Stereo Calibration failed! Exiting.")
            self.log.pLogErr("")
    
    # Function: to rectify calibration images and compute disparity maps for each image pair
    #
    def rectifyCalibImages(self):
        # List calibration images:
        images = glob.glob(self.sInputFilePath+"*")
        # Check that calibration has been completed 
        if self.bFlagCalibrationComplete:
            self.log.pLogMsg('')
            self.log.pLogMsg('Run [IMAGE RECTIFICATION] on calibration image set')
            self.log.pLogMsg('')
            self.log.pLogMsg(str(len(images))+' image pairs found. Start rectification.')
            iIndex = 0
            for fname in tqdm(images):
                # Load image
                dual_img = cv.imread(fname)
                # Convert to grayscale
                dual_img = cv.cvtColor(dual_img, cv.COLOR_BGR2GRAY)
                
                (h, w) = dual_img.shape
                # Compute separator x position
                w2 = int(w/2)
                # Create list to store images
                img_list = []
                # Append left image 
                img_list.append(dual_img[ 0:h , 0:w2 ])
                # Append right image 
                img_list.append(dual_img[ 0:h , w2:w ])
                # Create list with rectified images 
                rect_img_list = []
                
                for index, image in enumerate(img_list):
                    h,  w = image.shape[:2]
                    if index == 1:
                        aKmat = self.stereoCalibData.K1
                        aDist = self.stereoCalibData.D1
                    else:
                        aKmat = self.stereoCalibData.K2
                        aDist = self.stereoCalibData.D2

                    newcameramtx, roi = cv.getOptimalNewCameraMatrix(aKmat,
                                                                    aDist,
                                                                    (w, h), 1,
                                                                    (w, h))

                    if self.bFlagUseRemapping:
                        # undistort
                        mapx, mapy = cv.initUndistortRectifyMap(aKmat,
                                                                aDist,
                                                                None,
                                                                newcameramtx,
                                                                (w, h),
                                                                cv.CV_32FC1)
                        
                        dst = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)
                    else:
                        # undistort
                        dst = cv.undistort(image,
                                        self.aKmat,
                                        self.aDist,
                                        None,
                                        newcameramtx)

                    # crop the image
                    x, y, w, h = roi
                    if self.bFlagCropRectifImages:
                        dst = dst[y:y+h, x:x+w]
                        
                    # Append rectified image to list 
                    rect_img_list.append(dst)

                frameToDisplay = cv.hconcat([rect_img_list[0], rect_img_list[1]])
                disparity_image = self._computeDepthMap(rect_img_list[0], rect_img_list[1]) 
                # Print status
                fileName = str(iIndex) + '_rectified'
                fileNameDispMap = str(iIndex) + '_dispMap'
                cv.imwrite(self.sRecifiedImgPath + fileName+".png", frameToDisplay)
                cv.imwrite(self.sDisparityMapsPath + fileNameDispMap+".png", disparity_image)
                iIndex = iIndex + 1
        else:
            self.log.pLogErr('Rectification aborted. No calibration data.')
         

        
    #---------------------------------------------------------------------------
    #
    #   [ Private Functions ]
    #
    #---------------------------------------------------------------------------
    def _monoCalibrate(self):
        # ----------------------------------------------------------------------
        # run (Mono) Camera calibration
        # ----------------------------------------------------------------------
        CalibCritera = (cv.TERM_CRITERIA_EPS +
            cv.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
        
        # Create Image and Object Point array from imageData
        self._createImageAndObjectPointArray()
        
        self.log.pLogMsg(' ')
        self.log.pLogMsg(' [MONO CALIBRATE] ==> [LEFT CAM]')
        self.log.pLogMsg(' ')
        
        # Create camera calibration parameters
        (   left_success,
            self.LeftCamera.Kmat,
            self.LeftCamera.Dvec,
            self.LeftCamera.rvec,
            self.LeftCamera.tvec
        ) = cv.calibrateCamera( self.objpoints,
                                self.leftImgpoints,
                                self.imageSize,
                                None,
                                None,
                                criteria=CalibCritera)
            
        (   right_success,
            self.RightCamera.Kmat,
            self.RightCamera.Dvec,
            self.RightCamera.rvec,
            self.RightCamera.tvec 
        ) = cv.calibrateCamera( self.objpoints,
                                self.rightImgpoints,
                                self.imageSize,
                                None,
                                None,
                                criteria=CalibCritera)
            
        # Distribute rvecs and tvecs to the respective imageData's
        for ii, index in enumerate( self.aImageListToObjPointList) :
            self.aImageList[index].left_rvec  = self.LeftCamera.rvec[ii]
            self.aImageList[index].left_tvec  = self.LeftCamera.tvec[ii]
            self.aImageList[index].right_rvec = self.RightCamera.rvec[ii]
            self.aImageList[index].right_tvec = self.RightCamera.tvec[ii]
                    
        # Log success
        self.log.pLogMsg("")
        self.log.pLogMsg('Mono Camera calibration finished.')
        self.log.pLogMsg("")
        self.log.pLogMsg("[LEFT  CAM] Intrinsic Matrix: "+str(self.LeftCamera.Kmat))
        self.log.pLogMsg("")
        self.log.pLogMsg("[LEFT  CAM] Distortion Coefficients: "+str(self.LeftCamera.Dvec))
        self.log.pLogMsg("")
        self.log.pLogMsg("[RIGHT CAM] Intrinsic Matrix: "+str(self.RightCamera.Kmat))
        self.log.pLogMsg("")
        self.log.pLogMsg("[RIGHT CAM] Distortion Coefficients: "+str(self.RightCamera.Dvec))
        self.log.pLogMsg("")
      
    def _createAcircleObjectPoints(self):
        xx = 0
        yy = 0
        zz = 0
        nrCircles = self.boardSize[1] * self.boardSize[0]
        self.objp = np.zeros((nrCircles, 3), np.float32)
        obj_index = 0 
        
        # loop columns
        for ii in range(self.boardSize[1] ):
            for jj in range( self.boardSize[0] ):
                self.objp[obj_index] = (xx, yy, zz)
                yy = yy + self.objectSize
                obj_index = obj_index + 1
            
            xx = xx + self.objectSize/2
            if (ii % 2) == 0:
                yy = self.objectSize / 2
            else:
                yy = 0
                
        TBD=True
        
    def _createChessboardObjectPoints(self):
        self.objp = np.zeros((self.boardSize[1]*self.boardSize[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.boardSize[0],
                            0:self.boardSize[1]].T.reshape(-1, 2) * self.objectSize
        
    # Function: Create Image point array for all valid calibration images 
    def _createImageAndObjectPointArray(self):
        self.leftImgpoints  = []
        self.rightImgpoints = []
        self.objpoints      = []
        # Init list to point from image/object point arrays to the correct list index
        # in aImageList 
        # This is needed to distribute r/t vecs after calibration/re-calibration
        self.aImageListToObjPointList = []
        for ii, imageData in enumerate( self.aImageList ):
            if imageData.isPatternFound == True:
                self.leftImgpoints.append(  imageData.leftImagePoints )
                self.rightImgpoints.append( imageData.rightImagePoints )
                self.objpoints.append( self.objp )
                self.aImageListToObjPointList.append(ii)
                
    # Create blob detector to support assymetric circle detection  
    # TODO: currently not used. Assess if needed.       
    def _createBlobDetector(self):
        params = cv.SimpleBlobDetector_Params()
        
        params.minThreshold = 50
        params.maxThreshold = 220
        
        params.filterByColor = True
        params.blobColor = 0
        
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 5000
        
        params.filterByCircularity = False  
        params.minCircularity = 0.7
        params.maxCircularity = 3.4e38
        
        params.filterByConvexity = True
        params.minConvexity = 0.95
        params.maxConvexity = 3.4e38
        
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.maxInertiaRatio = 3e38
        
        self.blobDetector = cv.SimpleBlobDetector_create(params)
        
    def _saveResults(self):
        fileStorage = cv.FileStorage()
        if self.bFlagCalibrationComplete:
            fileStorage.open((self.sParameterFilePath+"stereo_calibration.yaml"), cv.FileStorage_WRITE)
            
            self.log.pLogMsg('> Saving stereo-calibration results to file.')
            
            fileStorage.write('K1', self.stereoCalibData.K1)
            fileStorage.write('D1',  self.stereoCalibData.D1)
            fileStorage.write('K2',  self.stereoCalibData.K2)
            fileStorage.write('D2',  self.stereoCalibData.D2)
            fileStorage.write('R',  self.stereoCalibData.R)
            fileStorage.write('T',  self.stereoCalibData.T)
            fileStorage.write('E',  self.stereoCalibData.E)
            fileStorage.write('F',  self.stereoCalibData.F)
            fileStorage.write('R1',  self.stereoCalibData.R1)
            fileStorage.write('R2',  self.stereoCalibData.R1)
            fileStorage.write('P1',  self.stereoCalibData.P1)
            fileStorage.write('P2',  self.stereoCalibData.P2)
            fileStorage.write('Q',  self.stereoCalibData.Q)    
            
            fileStorage.release()
            
    def _computeDepthMap(self, imgL, imgR):
        # SGBM Parameters -----------------
        # wsize default 3; 5; 7 for SGBM reduced size image; 
        # 15 for SGBM full size image (1300px and above); 
        window_size = 5

        left_matcher = cv.StereoSGBM_create(
            minDisparity=-1,
            numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=window_size,
            P1=8 * 3 * window_size,
            # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size,
            disp12MaxDiff=12,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=32,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )
        right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.3
        visual_multiplier = 6

        wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)

        wls_filter.setSigmaColor(sigma)
        displ = left_matcher.compute(imgL, imgR)  
        dispr = right_matcher.compute(imgR, imgL)  
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  

        filteredImg = cv.normalize(src=filteredImg, 
                                   dst=filteredImg, 
                                   beta=0, alpha=255, 
                                   norm_type=cv.NORM_MINMAX)
        
        filteredImg = np.uint8(filteredImg)

        return filteredImg
    
    def _calculateTotalReprojErrorPerImage(self):
        for ii, imageData in enumerate( self.aImageList ):
            if imageData.isPatternFound == True:
                imgpoints2_left, _ = cv.projectPoints(   self.objp,
                                                        imageData.left_rvec,
                                                        imageData.left_tvec,
                                                        self.stereoCalibData.K1,
                                                        self.stereoCalibData.D1)
                
                imgpoints2_right, _ = cv.projectPoints(   self.objp,
                                                        imageData.right_rvec,
                                                        imageData.right_tvec,
                                                        self.stereoCalibData.K2,
                                                        self.stereoCalibData.D2)
                error_left = cv.norm(imageData.leftImagePoints,
                                imgpoints2_left,
                                cv.NORM_L2)/len(imgpoints2_left)
                
                error_right = cv.norm(imageData.rightImagePoints,
                                imgpoints2_right,
                                cv.NORM_L2)/len(imgpoints2_right)
                
                reprojErrorArray_left  = np.absolute( np.array(imageData.leftImagePoints) - np.array(imgpoints2_left) )
                reprojErrorArray_right = np.absolute( np.array(imageData.rightImagePoints) - np.array(imgpoints2_right) )
                
                maxErrorPerImage  = np.max( np.array( reprojErrorArray_left  ) )
                minErrorPerImage  = np.min( np.array( reprojErrorArray_left ) )
                sdtvErrorPerImage = np.std( np.array(reprojErrorArray_left) )
                
                self.aImageList[ii].reprojErrorArray        = reprojErrorArray_left
                self.aImageList[ii].averageReprojError_left = error_left
                self.aImageList[ii].maxReprojError_left     = maxErrorPerImage
                self.aImageList[ii].minReprojError_left     = minErrorPerImage
                self.aImageList[ii].stdReprojError_left     = sdtvErrorPerImage
                
                maxErrorPerImage  = np.max( np.array( reprojErrorArray_right  ) )
                minErrorPerImage  = np.min( np.array( reprojErrorArray_right ) )
                sdtvErrorPerImage = np.std( np.array(reprojErrorArray_right) )
                
                self.aImageList[ii].reprojErrorArray         = reprojErrorArray_right
                self.aImageList[ii].averageReprojError_right = error_right
                self.aImageList[ii].maxReprojError_left      = maxErrorPerImage
                self.aImageList[ii].minReprojError_left      = minErrorPerImage
                self.aImageList[ii].stdReprojError_left      = sdtvErrorPerImage

    # Function create terminal print out with average reprojection error per valid calibration image
    def _showReprojectionError(self):
        self.log.pLogMsg("Reprojection error per image:")
        self._calculateTotalReprojErrorPerImage()
        aListReprojError_left   = self._returnAveragReprojErrPerImage("left")
        aListReprojError_right  = self._returnAveragReprojErrPerImage("right")
        maxAverageError_left  = max( aListReprojError_left )
        maxAverageError_right = max( aListReprojError_right )
        for imageData in self.aImageList:
            if imageData.isPatternFound == True:
                percReprojError_left  = imageData.averageReprojError_left  / maxAverageError_left  * 100 
                percReprojError_right = imageData.averageReprojError_right / maxAverageError_right * 100 
                
                sBarString_left  = self._calculateReprojBarString(percReprojError_left)
                sBarString_right = self._calculateReprojBarString(percReprojError_right)
                
                self.log.pLogMsg('#{:5.0f}'.format(imageData.imageIndex) + " [LEFT]  " + sBarString_left 
                                 +' - average: '+'{:.5f}'.format(imageData.averageReprojError_left)
                                 +' , min  '+'{:.5f}'.format(imageData.minReprojError_left)
                                 +' , max  '+'{:.5f}'.format(imageData.maxReprojError_left)
                                 +' , std  '+'{:.5f}'.format(imageData.stdReprojError_left)
                                )
                
                self.log.pLogMsg('#{:5.0f}'.format(imageData.imageIndex) + " [RIGHT] " + sBarString_right 
                                 +' - average: '+'{:.5f}'.format(imageData.averageReprojError_right)
                                 +' , min  '+'{:.5f}'.format(imageData.minReprojError_right)
                                 +' , max  '+'{:.5f}'.format(imageData.maxReprojError_right)
                                 +' , std  '+'{:.5f}'.format(imageData.stdReprojError_right)
                                )
    # Function to create string to visualize reprojection error 
    # e.g. |======    |
    #      |==        |
    def _calculateReprojBarString(self, percReprojError):
        # Init Bar with divider
        sBarString = "|"
        # Define max length of the bar
        # Equals number of characters that define 100 percent 
        barLength=50
        # calculate bar threshold in barLength space
        nrBars = round( (percReprojError / 100 ) * barLength)
        for ii in range(barLength):
            if ii < nrBars:
                # If within range add marker 
                sBarString = sBarString + "="
            else:
                # If outside range add space
                sBarString = sBarString + " "
        # Finish bar string with end divider
        sBarString = sBarString + "|"
        return sBarString
    
    # Functions returns a list of average reprojection error per image - only for
    # the valid images!   
    def _returnAveragReprojErrPerImage(self, camera):
        averageReprojList = []
        for imageData in self.aImageList:
            if imageData.isPatternFound == True:
                if camera is "left":   
                    averageReprojList.append(imageData.averageReprojError_left)
                else:
                    averageReprojList.append(imageData.averageReprojError_right)
        return averageReprojList