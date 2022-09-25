#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class CameraData():
    
    # Camera Name 
    CameraName = ""
    # Intrinsic Camera Matrix 
    Kmat = []
    # Distortion Coefficient 
    Dvec = []
    # Image size [ width , height ] [px]
    imgSize = []
    
    rvec = []
    tvec = []
    
    
    # Max Reprojection Error per image [px]
    maxReprojError = 0
    # Array of reprojection error per calibration image [px]
    aReprojErrors = []
    
    # >> Internal status flags
    # Flag: Mono camera calibration completed 
    isMonoCalibrationCompleted   = False 
    # Flag: Stereo camera calibration completed
    isStereoCalibrationCompleted = False
    
    imgPoints = []
    objPoints = []
    
    def __init__(self, CameraName):
        self.CameraName = CameraName 
        
    def setMonoCalibrated(self):
        self.isMonoCalibrationCompleted = True
        
    def setStereoCalibrated(self):
        self.isStereoCalibrationCompleted = True
        
    def saveToFile(self):
        TBD = True
        
    def readCalibrationFromFile(self):
        TBD = True
        
    def updateReprojectionStatistics(self):
        TBD = True
        # np.max() np.min() np.mean() np.std()
        
    