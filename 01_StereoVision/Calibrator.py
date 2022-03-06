#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:43:57 2022

@author: x
"""
import cv2 as cv
from tqdm import tqdm
import yaml
import glob
import numpy as np


class Calibrator:
    #--------------------------------------------------------------------------
    # Set global variables
    #--------------------------------------------------------------------------
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Internal state flags
    # Flag calibration run completed
    bFlagCalibrationComplete= False
    
    # Settings flags 
    bFlagEnableImgScaling   = False
    bFlagSaveMarkedImages   = False
    bFlagSaveScaledImages   = False
    bFlagUseRemapping       = False
    
    # File Paths 
    sInputFilePath          = ''
    sProcessedImagePath     = ''
    sScaledImgPath          = ''
    sParameterFilePath      = ''
    sRecifiedImgPath        = ''
    
    # Process images at scale: 
    scale_percent   = 35
    
    # Chessboard pattern dimension
    boardSize       = (0,0)
    
    # Number of images in input set 
    iNrImagesInSet  = 0
    
    aSuccIndexList = []
    
    # Camera claibration paramters 
    ret   = 0  
    aKmat = []
    aDist = []
    rvecs = []
    tvecs = []
    totalReprojectionError = 0 
    
    aListReprojError = []

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def setEnableRemapping(self, bFlagUseRemapping):
        self.bFlagUseRemapping = bFlagUseRemapping
    
    def setEnableImgScaling(self, bFlagEnableImgScaling):
        self.bFlagEnableImgScaling = bFlagEnableImgScaling
        
    def setEnableMarkedImages(self, bFlagSaveMarkedImages):
        self.bFlagSaveMarkedImages = bFlagSaveMarkedImages
    
    def setEnableSaveScaledImages(self, bFlagSaveScaledImages):
        self.bFlagSaveScaledImages = bFlagSaveScaledImages
        
    def getObjectPoints(self):
        return self.objpoints
    
    def __init__(self, sInputFilePath, sProcessedImagePath, sScaledImgPath, 
                 sParameterFilePath, sRecifiedImgPath, boardSize, log):
        self.log = log
        self.sInputFilePath         = sInputFilePath
        self.sProcessedImagePath    = sProcessedImagePath
        self.sScaledImgPath         = sScaledImgPath
        self.sParameterFilePath     = sParameterFilePath
        self.sRecifiedImgPath       = sRecifiedImgPath
        
        self.boardSize = boardSize
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((boardSize[1]*boardSize[0],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:boardSize[0],0:boardSize[1]].T.reshape(-1,2)
    
    def calibrate(self):
        # List calibration images:
        images = glob.glob(self.sInputFilePath+"*")
        imageIndex = 0
        self.iNrImagesInSet = 0
        # Reset success index list
        self.aSuccIndexList = []
        
        for fname in tqdm(images):
            # Load image 
            img         = cv.imread(fname)
            # Convert image to greyscale 
            gray        = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Scale image 
            if self.bFlagEnableImgScaling:
                width  = int(gray.shape[1] * self.scale_percent / 100 )
                height = int(gray.shape[0] * self.scale_percent / 100 )
                dsize = (width, height)
                outputImage = cv.resize(gray, dsize)
                outputColor = cv.resize(img, dsize)
            else:
                outputImage = gray
                outputColor = img

            # Print status
            fileName    = fname.split("/")
            fileName    = fileName[-1]
            self.log.pLogMsg('Load: ' + fileName)
            self.log.pLogMsg("Image loaded, Analizying...")

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(outputImage, 
                                                    self.boardSize,
                                                    None)
            

            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                corners2    = cv.cornerSubPix(outputImage,corners, 
                                              (11,11), 
                                              (-1,-1), 
                                              self.criteria)
                
                # Update success state 
                self.bFlagCalibrationComplete   =  True
                self.aSuccIndexList.append(imageIndex)
                
                self.iNrImagesInSet = self.iNrImagesInSet + 1
                # Save scaled images with drawn corners
                if self.bFlagSaveMarkedImages:
                    cv.drawChessboardCorners(outputColor, 
                                             self.boardSize, 
                                             corners2, 
                                             ret)
                    savePath = self.sProcessedImagePath + fileName
                    self.log.pLogMsg("Result images saved to: "+savePath)
                    cv.imwrite(savePath, outputColor)
                if self.bFlagSaveScaledImages:
                    savePath = self.sScaledImgPath + fileName
                    self.log.pLogMsg("Scaled images saved to: "+savePath)
                    cv.imwrite(savePath, outputImage)
            else:
                self.log.pLogMsg("")
                self.log.pLogMsg(str(imageIndex)+")  No chessboard found.")
                self.log.pLogMsg("")
            imageIndex = imageIndex + 1

        # self.log.pLogMsg('>> Pattern found in '+str(self.iNrImagesInSet) + '/'+str(imageIndex)+
        #                  " images. "+
        #                  str(self.iNrImagesInSet/imageIndex * 100)+
        #                  " percent yield. ")  
        self.log.pLogMsg('>> Pattern found in '
                         +str(self.iNrImagesInSet) 
                         + '/'+str(imageIndex)
                         +" images. %.2f percent yield" %
                         (self.iNrImagesInSet/imageIndex * 100) )  
        #----------------------------------------------------------------------
        # run Camera calibration 
        #----------------------------------------------------------------------
        if self.bFlagCalibrationComplete:
            # Create camera calibration parameters 
            self.ret, self.aKmat, self.aDist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, 
                                                              self.imgpoints, 
                                                              outputImage.shape[::-1], 
                                                              None, 
                                                              None)   
            # Log success
            self.log.pLogMsg("")
            self.log.pLogMsg('Camera calibration finished.')
            self.log.pLogMsg("")
            self.log.pLogMsg("Intrinsic Matrix: "+str(self.aKmat))
            self.log.pLogMsg("Distortion Coefficients: "+str(self.aDist))
            # Calculate reprojection error
            self._calculateTotalReprojError()
            self.log.pLogMsg( "Total reprojection error: " + 
                            str(self.totalReprojectionError) )
        else:
            self.log.pLogMsg("")
            self.log.pLogErr('Calibration failed (No chessboard found)')
            self.log.pLogMsg("")
            
    def showReprojectionError(self):
        self.log.pLogMsg("Reprojection error per image:")
        self.calcReproErrPerImage()
        for iImg in range(len(self.aListReprojError)):
            self.log.pLogMsg('#'+str(self.aSuccIndexList[iImg])+' - %.5f ' % self.aListReprojError[iImg] )
            
    def calcReproErrPerImage(self):
        # Flush Reprojection error list 
        self.aListReprojError = []
        # Check if object points are populated 
        if len(self.objpoints) != 0:
            for iImg in range(len(self.objpoints)):
                self.aListReprojError.append(self._calculateReprojectionError(iImg))
        else:
            self.log.pLogMsg('Object list is empty. Aborting.')

    # Function: Save calibration parameters to yaml file 
    def saveResults(self):
        if self.bFlagCalibrationComplete:
            # Compile data structure 
            data = {'camera_matrix': np.asarray(self.aKmat).tolist(), 
                    'dist_coeff': np.asarray(self.aDist).tolist()}
            with open(self.sParameterFilePath+"calibration.yaml", "w") as f:
                yaml.dump(data, f)

            self.log.pLogMsg('Camera parameters saved.')

        else:
            self.log.pLogMsg("")
            self.log.pLogErr('Calibration has not been completed. Run calibration first')
            self.log.pLogMsg("")
            
    def _calculateTotalReprojError(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], 
                                             self.rvecs[i], 
                                             self.tvecs[i], 
                                             self.aKmat, 
                                             self.aDist)
            error = cv.norm(self.imgpoints[i], 
                            imgpoints2, 
                            cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        self.totalReprojectionError = mean_error/len(self.objpoints)
    
    # Function: calculate reprojection error per image 
    def _calculateReprojectionError(self, iImgIndex):
        error = 0
        if self.bFlagCalibrationComplete:

            imgpoints2, _ = cv.projectPoints(self.objpoints[iImgIndex], 
                                             self.rvecs[iImgIndex], 
                                             self.tvecs[iImgIndex], 
                                             self.aKmat, 
                                             self.aDist)
            error = cv.norm(self.imgpoints[iImgIndex], 
                            imgpoints2, 
                            cv.NORM_L2)/len(imgpoints2)
        return error
    
    def rectify(self):
        # List calibration images:
        images = glob.glob(self.sInputFilePath+"*")
        if self.bFlagCalibrationComplete:
            self.log.pLogMsg('Run image rectification:')
            iIndex = 0 
            for fname in tqdm(images):
                # Load image 
                img         = cv.imread(fname)
                h,  w       = img.shape[:2]
    
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.aKmat, 
                                                                 self.aDist, 
                                                                 (w,h), 1, 
                                                                 (w,h))
    
                if self.bFlagUseRemapping:
                    # undistort
                    mapx, mapy = cv.initUndistortRectifyMap(self.aKmat, 
                                                            self.aDist, 
                                                            None, 
                                                            newcameramtx, 
                                                            (w,h), 
                                                            5)
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
                # dst = dst[y:y+h, x:x+w]
    
                # Print status
                fileName    = str(iIndex) + '_rectified'
                #print("")
                #print('Save Result file : ' + fileName)
                cv.imwrite(self.sRecifiedImgPath + fileName+".png", dst)
                iIndex = iIndex + 1
        else:
            self.log.pLogErr('Rectification aborted. No calibration data.')
    
    