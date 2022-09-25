#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:43:57 2022

@author: x
"""
import cv2 as cv
from tqdm import tqdm
import glob
import numpy as np

from util.CameraMetaData import CameraData
from util.CalibImageData import CalibImageData


class MonoCalibrator:
    # --------------------------------------------------------------------------
    # Set global variables
    # --------------------------------------------------------------------------
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Internal state flags
    # Flag calibration run completed
    bFlagCalibrationComplete = False

    # Settings flags
    bFlagEnableImgScaling = False
    bFlagSaveMarkedImages = False
    bFlagSaveScaledImages = False
    bFlagUseRemapping     = False

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
    rvecs = []
    tvecs = []
    totalReprojectionError = 0

    rectificationAlpha = 0

    imageSize = []

    subPixCriteria = (cv.TERM_CRITERIA_EPS +
                      cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

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
                 sParameterFilePath, sRecifiedImgPath, boardSize, objectSize,
                 sPatternType, log):
        self.log = log
        self.sInputFilePath         = sInputFilePath
        self.sProcessedImagePath    = sProcessedImagePath
        self.sScaledImgPath         = sScaledImgPath
        self.sParameterFilePath     = sParameterFilePath
        self.sRecifiedImgPath       = sRecifiedImgPath
        self.sPatternType           = sPatternType

        self.boardSize              = boardSize
        
        self.objectSize             = objectSize

        self.aStereoImagePairs      = []

        self.LeftCamera  = CameraData("Left_Camera")
        self.RightCamera = CameraData("Right_Camera")

        # Prepare object points
        if self.sPatternType == "acircles":
            self._createAcircleObjectPoints()
        elif self.sPatternType == "chessboard":
            self._createChessboardObjectPoints()
        else:
            self.pLogErr("Calibration board pattern type not valid!")
        
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
    # --------------------------------------------------------------------------
    #   >> Class functions Mono Calibration
    # --------------------------------------------------------------------------
    def calibrate(self):
        # List calibration images:
        self.images = glob.glob(self.sInputFilePath+"*")
        imageIndex = 0
        self.iNrImagesInSet = 0
        # Reset success index list
        self.aSuccIndexList = []
        
        self.aImageList = []
        
        CalibCritera = (cv.TERM_CRITERIA_EPS +
                cv.TERM_CRITERIA_MAX_ITER, 300, 1e-6)

        for ii, fname in tqdm( enumerate( self.images ) ):
            # Load image
            img = cv.imread(fname)
            # Convert image to greyscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            outputImage = gray
            outputColor = img
            
            self.imageSize = outputImage.shape[::-1]

            # Print status
            fileName = str(imageIndex) + '_calib.png'
            self.log.pLogMsg('Load: ' + fileName)
            self.log.pLogMsg("Image loaded, Analizying...")
            patternFound = False

            if self.sPatternType is  "acircles":
                # Set flags for assymetric circle pattern detection
                readFlags = cv.CALIB_CB_CLUSTERING
                readFlags |= cv.CALIB_CB_ASYMMETRIC_GRID
                
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                
                # Create blob detector to support finding assymetric 
                # circles pattern
                # TODO: see if needed
                # self.createBlobDetector()
                
                patternFound, corners = cv.findCirclesGrid( outputImage, 
                                                    self.boardSize, 
                                                    None,  
                                                    flags=readFlags)
            elif self.sPatternType is  "chessboard":
                readFlags = cv.CALIB_CB_ADAPTIVE_THRESH
                readFlags |= cv.CALIB_CB_FILTER_QUADS
                readFlags |= cv.CALIB_CB_NORMALIZE_IMAGE
                # Find the chess board corners
                patternFound, corners = cv.findChessboardCorners(outputImage,
                                                        self.boardSize,
                                                        flags=readFlags)

            # Create imageData instance 
            calibImageData = CalibImageData()
            
            # If found, add object points, image points (after refining them)
            if patternFound == True:
                
                # If Chessboard calibration -> run cornerSubPix refinement
                if self.sPatternType is  "chessboard":
                    corners2 = cv.cornerSubPix(outputImage, corners,
                                            (11, 11),
                                            (-1, -1),
                                            self.subPixCriteria)
                
                # Update success state
                self.bFlagCalibrationComplete = True
                self.aSuccIndexList.append(imageIndex)

                self.iNrImagesInSet = self.iNrImagesInSet + 1
                # Save scaled images with drawn corners
                if self.bFlagSaveMarkedImages:
                    cv.drawChessboardCorners(outputColor,
                                             self.boardSize,
                                             corners,
                                             patternFound)
                    savePath = self.sProcessedImagePath + fileName
                    self.log.pLogMsg("Result images saved to: "+savePath)
                    cv.imwrite(savePath, outputColor)
                if self.bFlagSaveScaledImages:
                    savePath = self.sScaledImgPath + fileName
                    self.log.pLogMsg("Scaled images saved to: "+savePath)
                    cv.imwrite(savePath, outputImage)
            else:
                self.log.pLogMsg(str(imageIndex)+")  [NO CALIBRATION PATTERN FOUND].")
            
            # Set Values:
            calibImageData.imageFilePath    = fname
            calibImageData.imageIndex       = ii
            calibImageData.isPatternFound   = patternFound
            calibImageData.imagePoints      = corners
            # Append image data to list
            self.aImageList.append(calibImageData)
            
            imageIndex = imageIndex + 1


        self.log.pLogMsg('>> Pattern found in '
                         + str(self.iNrImagesInSet)
                         + '/'+str(imageIndex)
                         + " images. %.2f percent yield" %
                         (self.iNrImagesInSet/imageIndex * 100))
        # ----------------------------------------------------------------------
        # run (Mono) Camera calibration
        # ----------------------------------------------------------------------
        if self.bFlagCalibrationComplete:
            # Create Image and Object Point array from imageData
            self.createImageAndObjectPointArray()
            
            self.log.pLogMsg(' ')
            self.log.pLogMsg(' [MONO CALIBRATE].')
            self.log.pLogMsg(' ')
            # Create camera calibration parameters
            (   self.ret,
                self.aKmat,
                self.aDist,
                self.rvecs,
                self.tvecs ) = cv.calibrateCamera(  self.objpoints,
                                                    self.imgpoints,
                                                    self.imageSize,
                                                    None,
                                                    None,
                                                    criteria=CalibCritera)
                
            # Distribute rvecs and tvecs to the respective imageData's
            for ii, index in enumerate( self.aImageListToObjPointList) :
                self.aImageList[index].rvec = self.rvecs[ii]
                self.aImageList[index].tvec = self.tvecs[ii]
                        
            # Log success
            self.log.pLogMsg("")
            self.log.pLogMsg('Camera calibration finished.')
            self.log.pLogMsg("")
            self.log.pLogMsg("Intrinsic Matrix: "+str(self.aKmat))
            self.log.pLogMsg("Distortion Coefficients: "+str(self.aDist))
            
            # Calculate reprojection error
            self._calculateTotalReprojError()
            self.log.pLogMsg(' ')
            self.log.pLogMsg("Total reprojection error: " +
                             str(self.totalReprojectionError))
        else:
            self.log.pLogMsg("")
            self.log.pLogErr('Calibration failed (No chessboard found)')
            self.log.pLogMsg("")

    # Function: Create Image point array for all valid calibration images 
    def createImageAndObjectPointArray(self):
        self.imgpoints = []
        self.objpoints = []
        # Init list to point from image/object point arrays to the correct list index
        # in aImageList 
        # This is needed to distribute r/t vecs after calibration/re-calibration
        self.aImageListToObjPointList = []
        for ii, imageData in enumerate( self.aImageList ):
            if imageData.isPatternFound == True:
                self.imgpoints.append( imageData.imagePoints )
                self.objpoints.append( self.objp )
                self.aImageListToObjPointList.append(ii)
    
    # Create blob detector to support assymetric circle detection  
    # TODO: currently not used. Assess if needed.       
    def createBlobDetector(self):
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
            self.createImageAndObjectPointArray()
            # Create camera calibration parameters
            (   self.ret,
                self.aKmat,
                self.aDist,
                self.rvecs,
                self.tvecs) = cv.calibrateCamera(   self.objpoints,
                                                    self.imgpoints,
                                                    self.imageSize,
                                                    None,
                                                    None,
                                                    criteria=CalibCritera)
                
            # Distribute rvecs and tvecs to the respective imageData's
            for ii, index in enumerate( self.aImageListToObjPointList) :
                self.aImageList[index].rvec = self.rvecs[ii]
                self.aImageList[index].tvec = self.tvecs[ii]
                        
            # Log success
            self.log.pLogMsg("")
            self.log.pLogMsg('Camera calibration finished.')
            self.log.pLogMsg("")
            self.log.pLogMsg("Intrinsic Matrix: "+str(self.aKmat))
            self.log.pLogMsg("Distortion Coefficients: "+str(self.aDist))
            # Calculate reprojection error
            self._calculateTotalReprojError()
            self.log.pLogMsg("Total reprojection error: " +
                             str(self.totalReprojectionError))
        else:
            self.log.pLogMsg("")
            self.log.pLogErr('Re-calibration failed (CalibrationComplete flag is False)')
            self.log.pLogMsg("")

    def discardReprojectionOutliers(self, reprojErrThreshold):
        self.log.pLogMsg("")
        self.log.pLogMsg("Remove Outliers based on reprojection errors.")
        self.log.pLogMsg("Threshold: "+str(reprojErrThreshold))
        self.log.pLogMsg("")
        
        # Init counter to count discarded images
        counter = 0
        
        for ii, imageData in enumerate( self.aImageList ):
            if imageData.isPatternFound == True and imageData.averageReprojError > reprojErrThreshold:
                # Reprojection Error above limits -> Remove from list of valid images
                self.aImageList[ii].isPatternFound = False
                # Print out move:
                self.log.pLogMsg("Remove Image "+str(imageData.imageIndex)+" ("+str(imageData.averageReprojError)+")")
                counter = counter + 1
        self.log.pLogMsg("A total of  "+str(counter)+" have been removed.") 
        return counter   
        
    # Function create terminal print out with average reprojection error per valid calibration image
    def showReprojectionError(self):
        self.log.pLogMsg("Reprojection error per image:")
        self.calcReproErrPerImage()
        aListReprojError = self.returnAveragReprojErrPerImage()
        maxAverageError = max( aListReprojError )
        for imageData in self.aImageList:
            if imageData.isPatternFound == True:
                percReprojError = imageData.averageReprojError / maxAverageError * 100 
                sBarString = self.calculateReprojBarString(percReprojError)
                self.log.pLogMsg('#'+str(imageData.imageIndex) + " " + sBarString 
                                 +' - average: '+'{:.5f}'.format(imageData.averageReprojError)
                                 +' , min  '+'{:.5f}'.format(imageData.minReprojError)
                                 +' , max  '+'{:.5f}'.format(imageData.maxReprojError)
                                 +' , std  '+'{:.5f}'.format(imageData.stdReprojError)
                                )

    # Function to create string to visualize reprojection error 
    # e.g. |======    |
    #      |==        |
    def calculateReprojBarString(self, percReprojError):
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
    
    # Function: Calculate average reprojection error per image 
    def calcReproErrPerImage(self):
        # Check if object points are populated
        for ii, imageData in enumerate( self.aImageList ):
            if imageData.isPatternFound == True:
                # Calculate average reprojection error per image
                self.aImageList[ii].averageReprojError = (
                    self.calcAverageReporjErrorPerImage(imageData))
                # Caluclate reprojection error array for all corner points
                # per image 
                reprojArr = self.calcReprojErrorArrayPerImage(imageData)
                maxErrorPerImage  = np.max( np.array( reprojArr  ) )
                minErrorPerImage  = np.min( np.array( reprojArr ) )
                sdtvErrorPerImage = np.std( np.array(reprojArr) )
                self.aImageList[ii].reprojErrorArray = reprojArr
                self.aImageList[ii].maxReprojError   = maxErrorPerImage
                self.aImageList[ii].minReprojError   = minErrorPerImage
                self.aImageList[ii].stdReprojError   = sdtvErrorPerImage
         
    # Functions returns a list of average reprojection error per image - only for
    # the valid images!   
    def returnAveragReprojErrPerImage(self):
        averageReprojList = []
        for imageData in self.aImageList:
            if imageData.isPatternFound == True:
                averageReprojList.append(imageData.averageReprojError)
        return averageReprojList
            
    def saveResults(self):
        fileStorage = cv.FileStorage()
        if self.bFlagCalibrationComplete:
            fileStorage.open((self.sParameterFilePath+"calibration.yaml"), cv.FileStorage_WRITE)
            
            self.log.pLogMsg('Save calibration results to file.')
            
            fileStorage.write('K_MAT', self.aKmat)
            fileStorage.write('D_VEC', self.aDist)
            
            fileStorage.release()

    def _calculateTotalReprojError(self):
        mean_error = 0
        for imageData in self.aImageList:
            if imageData.isPatternFound == True:
                imgpoints2, _ = cv.projectPoints(   self.objp,
                                                    imageData.rvec,
                                                    imageData.tvec,
                                                    self.aKmat,
                                                    self.aDist)
                error = cv.norm(imageData.imagePoints,
                                imgpoints2,
                                cv.NORM_L2)/len(imgpoints2)
                mean_error += error
        self.totalReprojectionError = mean_error/len(self.objpoints)

    # Function: calculate reprojection error per image
    def calcAverageReporjErrorPerImage(self, imageData):
        error = 0
        # Check if calibration has been completed 
        if self.bFlagCalibrationComplete and imageData.isPatternFound == True:

            imgpoints2, _ = cv.projectPoints(self.objp,
                                             imageData.rvec,
                                             imageData.tvec,
                                             self.aKmat,
                                             self.aDist)
            error = cv.norm(imageData.imagePoints,
                            imgpoints2,
                            cv.NORM_L2)/len(imgpoints2)
        return error
    
    def calcReprojErrorArrayPerImage(self, imageData):
        # reprojErrorArray = []
        # if self.bFlagCalibrationComplete and imageData.isPatternFound == True:

        imgpoints2, _ = cv.projectPoints(self.objp,
                                            imageData.rvec,
                                            imageData.tvec,
                                            self.aKmat,
                                            self.aDist)
        reprojErrorArray = np.absolute( np.array(imageData.imagePoints) - np.array(imgpoints2) )
            
        return reprojErrorArray
        

    def rectify(self):
        # List calibration images:
        images = glob.glob(self.sInputFilePath+"*")
        if self.bFlagCalibrationComplete:
            self.log.pLogMsg('')
            self.log.pLogMsg('Run [IMAGE RECTIFICATION]]')
            self.log.pLogMsg('')
            iIndex = 0
            for fname in tqdm(images):
                # Load image
                img = cv.imread(fname)
                h,  w = img.shape[:2]

                newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.aKmat,
                                                                 self.aDist,
                                                                 (w, h), 1,
                                                                 (w, h))

                if self.bFlagUseRemapping:
                    # undistort
                    mapx, mapy = cv.initUndistortRectifyMap(self.aKmat,
                                                            self.aDist,
                                                            None,
                                                            newcameramtx,
                                                            (w, h),
                                                            cv.CV_32FC1)
                    
                    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
                else:
                    # undistort
                    dst = cv.undistort(img,
                                       self.aKmat,
                                       self.aDist,
                                       None,
                                       newcameramtx)

                # crop the image
                x, y, w, h = roi
                if self.bFlagCropRectifImages:
                    dst = dst[y:y+h, x:x+w]

                # Print status
                fileName = str(iIndex) + '_rectified'
                # print("")
                #print('Save Result file : ' + fileName)
                cv.imwrite(self.sRecifiedImgPath + fileName+".png", dst)
                iIndex = iIndex + 1
        else:
            self.log.pLogErr('Rectification aborted. No calibration data.')
            
