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
import os
import math
import json
from pathlib import Path

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
    bEnableCreateDispMapsFromCalibImgs = False

    # File Paths
    sInputFilePath      = ''
    sProcessedImagePath = ''
    sScaledImgPath      = ''
    sParameterFilePath  = ''
    sRecifiedImgPath    = ''

    # Process images at scale:
    scale_percent = 100

    # Chessboard pattern dimension
    # (rows, columns)
    boardSize = (0, 0)

    # Number of images in input set
    iNrImagesInSet = 0

    aSuccIndexList = []

    # Camera claibration paramters
    ret = 0
    aKmat = []
    aDist = []

    lmapx = []
    lmapy = []
    rmapx = []
    rmapy = []

    totalReprojectionError = 0

    rectificationAlpha = 0
    
    # Stereo camera class streaming mode constants
    # TODO: This should only be maintained here temporarily
    STEREOCAMERA_OBS_MODE_NONE          = 0
    STEREOCAMERA_OBS_MODE_DISPARITY_MAP = 1
    STEREOCAMERA_OBS_MODE_RECTIFIED     = 2
    STEREOCAMERA_OBS_MODE_RAW_IMG       = 3
    STEREOCAMERA_OBS_MODE_LAPLCE_IMG    = 4
    STEREOCAMERA_OBS_MODE_CALIBRATION   = 5

    imageSize = []
    # --------------------------------------------------------------------------
    #       OpenCV calibration setting parameters 
    # --------------------------------------------------------------------------
    subPixCriteria = (cv.TERM_CRITERIA_EPS +
                      cv.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    # Initialize stereo calibration flags
    stereoCalibFlags = 0
    # Set flag settings
    stereoCalibFlags |= cv.CALIB_FIX_FOCAL_LENGTH
    stereoCalibFlags |= cv.CALIB_FIX_INTRINSIC
    stereoCalibFlags |= cv.CALIB_USE_INTRINSIC_GUESS
    stereoCalibFlags |= cv.CALIB_ZERO_TANGENT_DIST

    stereoCalibCriteria = (cv.TERM_CRITERIA_MAX_ITER +
                        cv.TERM_CRITERIA_EPS, 300, 1e-6)
    
    rectificationFlags = cv.CALIB_ZERO_DISPARITY

    # alpha=-1 -> Let OpenCV optimize black parts.

    # alpha= 0 -> Rotate and cut the image so that there will be no black
    # parts. This option cuts the image so badly most of the time, that
    # you won’t have a decent high-quality image but worth to try.

    # alpha= 1 -> Make the transform but don’t cut anything.
    stereoCalibAlpha = 0
    # --------------------------------------------------------------------------
    # Restrict the number of calibration images to a maximum value:
    maxNrCalibrationImages = -1
    
    # Image list 
    # track reprojection errors and include flags
    aImageList = []

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
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
        self.log                    = log
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

        self.LeftCamera             = CameraData("Left_Camera")
        self.RightCamera            = CameraData("Right_Camera")
        
        self.stereoCalibData        = StereoCalibrationData(self.log)
        
        self.isUndistMapsLeftSaved  = False
        self.isUndistMapsRightSaved = False
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
        self.log.pLogMsg("[START STEREO CALIBRATION PROCEDURE]")
        self.log.pLogMsg("Calibration board pattern type        : {}".format(self.sPatternType))
        self.log.pLogMsg("Calibration board corner points       : {}".format(self.boardSize))
        self.log.pLogMsg("Calibration board corner distance [m] : {}".format(self.objectSize))
        self.log.pLogMsg("Raw calibration images path           : {}".format(self.sInputFilePath))
        self.log.pLogMsg("Save marked raw images                : {}".format(self.bFlagSaveMarkedImages))
        self.log.pLogMsg(self._createLargeSeparator())
        self.log.pLogMsg("")
        self.log.pLogMsg("[READ RAW CALIBRATION IMAGES]")
        self.log.pLogMsg("")
        self.log.pLogMsg(self._createLargeSeparator())
        
        
        # Opening JSON file
        self.log.pLogMsg("Read image set meta data file: ")
        with open((Path(self.sInputFilePath) / 'stereo_img_meta.json').absolute().as_posix()) as json_file:
            self.session_meta_data = json.load(json_file)
            
        if self.session_meta_data is None:
            self.log.pLogErr("")
            self.log.pLogErr("No calibration image set meta data file (stereo_img_meta.json) found.")
            self.log.pLogErr("Exiting.")
            self.log.pLogErr("")
            return 0
            
        # List calibration images:
        self.dual_images    = glob.glob((Path(self.sInputFilePath) / f"{self.session_meta_data['img_prefix']}*.png").absolute().as_posix())
        imageIndex          = 0
        self.iNrImagesInSet = 0
        # Reset success index list
        self.aSuccIndexList = []
        
        self.aImageList     = []
        
        if len(self.dual_images) == 0:
            self.log.pLogErr("")
            self.log.pLogErr("No calibration images found.")
            self.log.pLogErr("Exiting.")
            self.log.pLogErr("")
            return 0
        elif len(self.dual_images) != self.session_meta_data['number_of_images']:
            self.log.pLogErr("")
            self.log.pLogErr(f"Number of stereo frames found in folder ({len(self.dual_images)}) does not match number reported in meta data file ({self.session_meta_data['number_of_images']})")
            self.log.pLogErr("")
            input("Press Enter to accept and continue anyway...")
        elif self.session_meta_data['mode'] != self.STEREOCAMERA_OBS_MODE_CALIBRATION:
            self.log.pLogErr("")
            self.log.pLogErr(f"Input image meta data reports image capture mode that is not 'CALIBRATION' ({self.session_meta_data['mode']}) ")
            self.log.pLogErr("The selected mode images were caputered with is not intended for calibration purposes. ")
            self.log.pLogErr("")
            input("Press Enter to accept and continue anyway ...")
            

        for ii, fname in tqdm( enumerate( self.dual_images ) ):
            # Load dual image
            img = cv.imread(fname)

            if img is None: 
                self.log.pLogErr(f'Image loading returned: None')
                self.log.pLogErr(f'Image path: {fname}')
                exit(1)

            # If input is not grayscale -> Convert image to greyscale
            if(len(img.shape)<3):
                gray = img
            else:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            outputImage = gray
            outputColor = img
            
            # TODO: This could be replaced by using the respective value from the 
            #       meta data file
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

            if self.sPatternType ==  "acircles":
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
            elif self.sPatternType ==  "chessboard":
                readFlags  = cv.CALIB_CB_ADAPTIVE_THRESH
                readFlags |= cv.CALIB_CB_FILTER_QUADS
                readFlags |= cv.CALIB_CB_NORMALIZE_IMAGE
                # Find the chess board corners
                patternFound_left, corners_left   = cv.findChessboardCorners(leftImage,
                                                                            self.boardSize,
                                                                            flags=readFlags)
                patternFound_right, corners_right = cv.findChessboardCorners(rightImage,
                                                                            self.boardSize,
                                                                            flags=readFlags)

            # Create imageData instance 
            calibImageData = CalibImageData()
            
            # If found, add object points, image points (after refining them)
            if patternFound_left == True and patternFound_right == True:
                patternFound = True
                # If Chessboard calibration -> run cornerSubPix refinement
                corners2_left = cv.cornerSubPix(leftImage, 
                                                corners_left,
                                                (11, 11),
                                                (-1, -1),
                                                self.subPixCriteria)
                
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
                                             corners2_left,
                                             patternFound)
                    cv.drawChessboardCorners(rightImageColor,
                                             self.boardSize,
                                             corners2_right,
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

            if patternFound:
                # Set Values:
                calibImageData.imageFilePath        = fname
                calibImageData.imageIndex           = ii
                calibImageData.isPatternFound       = patternFound
                calibImageData.leftImagePoints      = corners2_left
                calibImageData.rightImagePoints     = corners2_right
                calibImageData.rawLeftImg           = leftImage
                calibImageData.rawRightImg          = rightImage
                # Append image data to list
                self.aImageList.append(calibImageData)
            
            imageIndex = imageIndex + 1

        self.log.pLogMsg(self._createLineSeparator())
        self.log.pLogMsg('>> Pattern found in '
                         + str(self.iNrImagesInSet)
                         + '/'+str(imageIndex)
                         + " images. %.2f percent yield" %
                         (self.iNrImagesInSet/imageIndex * 100))
        self.log.pLogMsg(self._createLineSeparator())

        self._monoCalibrate()
    
    # Function: Run stereo calibration and save images 
    def stereoCalibrate(self):
        self.log.pLogMsg(self._createLargeSeparator())
        self.log.pLogMsg("")
        self.log.pLogMsg("[RUN STEREO CALIBRATION]")
        self.log.pLogMsg("")
        self.log.pLogMsg(self._createLargeSeparator())

        # Check if any image pairs are in image buffer
        if len(self.aImageListToObjPointList) == 0:
            self.log.pLogErr(
                "List of valid stereo image pairs (pattern found) is empty. Exiting.")
            return False

        stereoCalibFlags = ( cv.CALIB_FIX_ASPECT_RATIO + 
                            cv.CALIB_ZERO_TANGENT_DIST +
                            cv.CALIB_USE_INTRINSIC_GUESS +
                            cv.CALIB_SAME_FOCAL_LENGTH +
                            cv.CALIB_FIX_PRINCIPAL_POINT )
        
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
         flags=stereoCalibFlags
         )

        # Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, 
        # the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted 
        # so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the 
        # rectified image is decimated and shifted so that all the pixels from the original images from the cameras 
        # are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an 
        # intermediate result between those two extreme cases. 
        self.stereoCalibAlpha = -1

        # Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set, the function makes the 
        # principal points of each camera have the same pixel coordinates in the rectified views. And if the flag 
        # is not set, the function may still shift the images in the horizontal or vertical direction (depending 
        # on the orientation of epipolar lines) to maximize the useful image area. 
        self.rectificationFlags = cv.CALIB_ZERO_DISPARITY

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
            newImageSize=self.imageSize,
            )

            print(roi_left)
            print(roi_right)

            self.log.pLogMsg("")
            self.log.pLogMsg("Stereo Rectification completed")
            self.log.pLogMsg("")
            
            # Print complete set of stereo calibration parameters
            self.stereoCalibData.printStereoCalibrationData(self.imageSize)
            self.log.pLogMsg(self._createLargeSeparator())

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
        images = glob.glob((Path(self.sInputFilePath) / f"{self.session_meta_data['img_prefix']}*.png").absolute().as_posix())
        
        # Check that calibration has been completed 
        if self.bFlagCalibrationComplete:
            self.log.pLogMsg('')
            self.log.pLogMsg('Run [IMAGE RECTIFICATION] on calibration image set')
            self.log.pLogMsg('')
            self.log.pLogMsg(str(len(images))+' image pairs found. Start rectification.')
            iIndex = 0
            isLeftRemapped  = False
            isRightRemapped = False

            for fname in tqdm(images):
                # Load image
                dual_img = cv.imread(fname)
                
                if dual_img is None:
                    self.log.pLogWrn(f'Loaded input image is None!')
                    self.log.pLogErr('Exiting')
                    self.log.pLogErr('')
                    exit(1)
                    
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

                    # The function computes the optimal new camera matrix based on the free scaling parameter. 
                    # By varying this parameter the user may retrieve only sensible pixels alpha=0, keep all the 
                    # original image pixels if there is valuable information in the corners alpha=1, or get something 
                    # in between. When alpha>0, the undistortion result will likely have some black pixels corresponding 
                    # to “virtual” pixels outside of the captured distorted image. The original camera matrix, distortion
                    # coefficients, the computed new camera matrix and the newImageSize should be passed to 
                    # InitUndistortRectifyMap to produce the maps for Remap.
                    newCamMatAlpha = 0
                    if index == 0 and not isLeftRemapped:
                        isLeftRemapped = True
                        newcameramtx, roil = cv.getOptimalNewCameraMatrix(self.stereoCalibData.K1,
                                                                         self.stereoCalibData.D1,
                                                                         (w, h), 
                                                                         newCamMatAlpha,
                                                                         (w, h))

                        self.lmapx, self.lmapy = cv.initUndistortRectifyMap(self.stereoCalibData.K1,
                                                                            self.stereoCalibData.D1,
                                                                            self.stereoCalibData.R1,
                                                                            newcameramtx,
                                                                            (w, h),
                                                                            cv.CV_32FC1)

                    if index == 1 and not isRightRemapped:
                        isRightRemapped = True
                        newcameramtx, roir = cv.getOptimalNewCameraMatrix(self.stereoCalibData.K2,
                                                                         self.stereoCalibData.D2,
                                                                         (w, h), 
                                                                         newCamMatAlpha,
                                                                         (w, h))

                        self.rmapx, self.rmapy = cv.initUndistortRectifyMap(self.stereoCalibData.K2,
                                                                self.stereoCalibData.D2,
                                                                self.stereoCalibData.R2,
                                                                newcameramtx,
                                                                (w, h),
                                                                cv.CV_32FC1)
                    
                    # Save undistoriton maps to file 
                    if not self.isUndistMapsLeftSaved:                                    
                        if index == 0:
                            self.log.pLogMsg('Saving left camera undistortion map -> caml_undistortion_map.tiff')
                            (caml_mapx, caml_mapy) = cv.convertMaps(map1=self.lmapx, map2=self.lmapy, dstmap1type=cv.CV_16SC2)
                            stacked = np.dstack([caml_mapx.astype(np.uint16), caml_mapy])
                            cv.imwrite(os.path.join(self.sParameterFilePath,"caml_undistortion_map.tiff"), stacked)
                            self.isUndistMapsLeftSaved = True
                    if not self.isUndistMapsRightSaved:
                        if index != 0:
                            self.log.pLogMsg('Saving right camera undistortion map -> camr_undistortion_map.tiff')
                            (camr_mapx, camr_mapy) = cv.convertMaps(map1=self.rmapx, map2=self.rmapy, dstmap1type=cv.CV_16SC2)
                            stacked = np.dstack([camr_mapx.astype(np.uint16), camr_mapy])
                            cv.imwrite(os.path.join(self.sParameterFilePath,"camr_undistortion_map.tiff"), stacked)
                            self.isUndistMapsRightSaved = True

                    # cv.CV_16SC2
                    # cv.CV_32FC1
                    if index == 0:
                        # Remap left image
                        dst = cv.remap(image, self.lmapx, self.lmapy, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT)
                        # crop the image
                        x, y, w, h = roil
                        if self.bFlagCropRectifImages:
                            dst = dst[y:y+h, x:x+w]
                    else:
                        # Remap right image 
                        dst = cv.remap(image, self.rmapx, self.rmapy, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT)
                        # crop the image
                        x, y, w, h = roir
                        if self.bFlagCropRectifImages:
                            dst = dst[y:y+h, x:x+w]

                        
                    # Append rectified image to list 
                    rect_img_list.append(dst)

                frameToDisplay = cv.hconcat([rect_img_list[0], rect_img_list[1]])
                # Number of evenly spaced horizontal lines             
                nrLines = 20
                enableDrawHorizontalLines = True

                # If enabled -> Draw horizontal lines to visually check row alignment
                if enableDrawHorizontalLines:
                    lineDistance= h/nrLines
                    (h, w) = frameToDisplay.shape
                    for line in range(nrLines):
                        lineY = int(line * lineDistance)
                        line_thickness = 1
                        cv.line(frameToDisplay, (0, lineY), (w, lineY), (0, 255, 0), thickness=line_thickness)
                    
                fileName = str(iIndex) + '_rectified'
                cv.imwrite(os.path.join(self.sRecifiedImgPath , fileName+".png"), frameToDisplay)

                if self.bEnableCreateDispMapsFromCalibImgs:
                    disparity_image = self._computeDepthMap(rect_img_list[0], rect_img_list[1]) 
                    fileNameDispMap = str(iIndex) + '_dispMap'
                    cv.imwrite(os.path.join(self.sDisparityMapsPath , fileNameDispMap+".png"), disparity_image)

                iIndex = iIndex + 1
        else:
            self.log.pLogErr('Rectification aborted. No calibration data.')
         

    # Function: re-run calibration with revised inclusion setting based on 
    #           reprojection errors achieved in previous iteration.
    # Note: Has to be run after self.calibrate!
    def recalibrate(self):
        self.log.pLogMsg(' ')
        self.log.pLogMsg(' [RECALIBRATE].')
        self.log.pLogMsg(' ')
        
        CalibCritera = (cv.TERM_CRITERIA_EPS +
        cv.TERM_CRITERIA_MAX_ITER, 300, 1e-6)
        # ----------------------------------------------------------------------
        # run (Mono) Camera calibration
        # ----------------------------------------------------------------------
        if self.bFlagCalibrationComplete:
            # Create Image and Object Point array from imageData
            self._createImageAndObjectPointArray()
            # Create camera calibration parameters
            self.stereoCalibrate()
                
            # Distribute rvecs and tvecs to the respective imageData's
            # for ii, index in enumerate( self.aImageListToObjPointList) :
            #     self.aImageList[index].rvec = self.rvecs[ii]
            #     self.aImageList[index].tvec = self.tvecs[ii]
        else:
            self.log.pLogMsg("")
            self.log.pLogErr('Re-calibration failed (CalibrationComplete flag is False)')
            self.log.pLogMsg("") 
    #---------------------------------------------------------------------------
    #
    #   [ Private Functions ]
    #
    #---------------------------------------------------------------------------
    def _monoCalibrate(self):
        # ----------------------------------------------------------------------
        # run (Mono) Camera calibration
        # ----------------------------------------------------------------------
        self.log.pLogMsg(self._createLargeSeparator())
        self.log.pLogMsg("")
        self.log.pLogMsg("[RUN MONO CALIBRATION]")
        self.log.pLogMsg("")
        self.log.pLogMsg(self._createLargeSeparator())
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

        self.log.pLogMsg(' ')
        self.log.pLogMsg(' [MONO CALIBRATE] ==> [Right CAM]')
        self.log.pLogMsg(' ')
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
                # Append summary of processed stereo bench properties:
        self.log.pLogMsg("[LEFT  CAM] Calibrated FoV in x [deg]: {}".format(math.degrees(2*np.arctan(self.imageSize[0]/(2*self.LeftCamera.Kmat[0,0])))))
        self.log.pLogMsg("[LEFT  CAM] Calibrated FoV in y [deg]: {}".format(math.degrees(2*np.arctan(self.imageSize[1]/(2*self.LeftCamera.Kmat[1,1])))))
        self.log.pLogMsg("")
        self.log.pLogMsg("[RIGHT CAM] Calibrated FoV in x [deg]: {}".format(math.degrees(2*np.arctan(self.imageSize[0]/(2*self.RightCamera.Kmat[0,0])))))
        self.log.pLogMsg("[RIGHT CAM] Calibrated FoV in y [deg]: {}".format(math.degrees(2*np.arctan(self.imageSize[1]/(2*self.RightCamera.Kmat[1,1])))))
      
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
    
    def discardReprojectionOutliers(self):
        """
        Discard image pair based on their summed reprojection error
        """
        self.log.pLogMsg("")
        self.log.pLogMsg("Remove Outliers based on reprojection errors.")
        self.log.pLogMsg("")
        
        # Init counter to count discarded images
        counter  = 0
        sigmaThr = 1
        aListReprojError_left   = self._returnAveragReprojErrPerImage("left")
        aListReprojError_right  = self._returnAveragReprojErrPerImage("right")
        left_stdv = np.std(aListReprojError_left)
        right_stdv = np.std(aListReprojError_right)
        left_mean = np.mean(aListReprojError_left)
        right_mean = np.mean(aListReprojError_right)
        
        leftThr  = left_mean  + sigmaThr * left_stdv
        rightThr = right_mean + sigmaThr * right_stdv
        
        for ii, imageData in enumerate( self.aImageList ):
            if ( imageData.isPatternFound == True 
                and (imageData.averageReprojError_left > leftThr 
                     or imageData.averageReprojError_right > rightThr))  :
                # Reprojection Error above limits -> Remove from list of valid images
                self.aImageList[ii].isPatternFound = False
                # Print out move:
                self.log.pLogMsg("Remove Image "+str(imageData.imageIndex))
                counter = counter + 1
        self.log.pLogMsg("A total of  "+str(counter)+" have been removed.") 
        return counter  

    def _calculateTotalReprojErrorPerImage(self):
        for ii, imageData in enumerate( self.aImageList ):
            if imageData.isPatternFound == True:
                # rvecs and tvecs should actually be computed by cv.stereoCalibrate. For some reason
                # that is beyond me that only seems to be an option in the cpp version and not with 
                # python (at least without touching the underlying cpp code). Hence this workaround for now.
                (retval, rvec_l, tvec_l,inliers) = cv.solvePnPRansac(self.objp, 
                                                                    imageData.leftImagePoints, 
                                                                    self.stereoCalibData.K1, 
                                                                    self.stereoCalibData.D1)
                imgpoints2_left, _ = cv.projectPoints(self.objp,
                                                    rvec_l,
                                                    tvec_l,
                                                    self.stereoCalibData.K1,
                                                    self.stereoCalibData.D1)
                
                (retval, rvec_r, tvec_r,inliers) = cv.solvePnPRansac(self.objp, 
                                                                    imageData.rightImagePoints, 
                                                                    self.stereoCalibData.K2, 
                                                                    self.stereoCalibData.D2)
                imgpoints2_right, _ = cv.projectPoints(self.objp,
                                                    rvec_r,
                                                    tvec_r,
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
                
                self.aImageList[ii].reprojErrorArray_left   = reprojErrorArray_left
                self.aImageList[ii].averageReprojError_left = error_left
                self.aImageList[ii].maxReprojError_left     = maxErrorPerImage
                self.aImageList[ii].minReprojError_left     = minErrorPerImage
                self.aImageList[ii].stdReprojError_left     = sdtvErrorPerImage
                self.aImageList[ii].reprojImgPoints_left    = imgpoints2_left
                
                maxErrorPerImage  = np.max( np.array(reprojErrorArray_right) )
                minErrorPerImage  = np.min( np.array(reprojErrorArray_right) )
                sdtvErrorPerImage = np.std( np.array(reprojErrorArray_right) )
                
                self.aImageList[ii].reprojErrorArray_right    = reprojErrorArray_right
                self.aImageList[ii].averageReprojError_right  = error_right
                self.aImageList[ii].maxReprojError_right      = maxErrorPerImage
                self.aImageList[ii].minReprojError_right      = minErrorPerImage
                self.aImageList[ii].stdReprojError_right      = sdtvErrorPerImage
                self.aImageList[ii].reprojImgPoints_right     = imgpoints2_right

    # Function create terminal print out with average reprojection error per valid calibration image
    def _showReprojectionError(self):
        self.log.pLogMsg("Reprojection error per image:")
        self.log.pLogMsg(self._createLineSeparator())

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
                self.log.pLogMsg(self._createLineSeparator())
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
    def _returnAveragReprojErrPerImage(self, sCameraId):
        averageReprojList = []
        for imageData in self.aImageList:
            if imageData.isPatternFound == True:
                if sCameraId == "left":   
                    averageReprojList.append(imageData.averageReprojError_left)
                elif sCameraId == 'right':
                    averageReprojList.append(imageData.averageReprojError_right)
                else:
                    self.pLogErr('sCameraId not valid.')
        return averageReprojList

    def _createLineSeparator(self):
        # Number of characters to match:
        cCount =  133
        sOut   = ''
        for iCounter in range(cCount):
            sOut = sOut + "-"
        return sOut

    def _createLargeSeparator(self):
        # Number of characters to match:
        cCount =  133
        sOut   = ''
        for iCounter in range(cCount):
            sOut = sOut + "="
        return sOut

    def drawReprojectedCornerPoints(self):

        self.log.pLogMsg('Save images with reprojected corner points: ')
        for iCounter, imageData in tqdm(enumerate(self.aImageList)):

            imgPointsLeft  = imageData.reprojImgPoints_left
            imgPointsRight = imageData.reprojImgPoints_right

            imgl_c = cv.cvtColor(imageData.rawLeftImg, cv.COLOR_GRAY2BGR)
            imgr_c = cv.cvtColor(imageData.rawRightImg, cv.COLOR_GRAY2BGR)

            cv.drawChessboardCorners(imgl_c,
                                        self.boardSize,
                                        imgPointsLeft,
                                        True)
            cv.drawChessboardCorners(imgr_c,
                                        self.boardSize,
                                        imgPointsRight,
                                        True)

            fileName = str(iCounter) + '_check.png'
            # TODO: If this turns out to be a useful feature there should be a dedicated folder to save the images 
            #       to. In any case the DisparityMap folder is only a dirty, temporary solution.ß
            savePath = self.sDisparityMapsPath + fileName
            dual_image = cv.hconcat([imgl_c, imgl_c])
            cv.imwrite(savePath, dual_image)
