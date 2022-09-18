#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 21:04:50 2022

@author: x
"""
import os
import datetime

from util.VisualOdometry import VisualOdometry

sTestFilePath               = "./test_data_set_01"
sCameraCalibrationFilePath  = "./camera_01_calibration/"

def main():
    testName = "test_set_01"
    
    bFlagRunLiveTest = False
    #-----------------------------------------------------------------------------------
    # [SETUP FOLDERS]
    #-----------------------------------------------------------------------------------
    # >> Setup folder structure 
    # Setup folder to save images to 
    dateTimeStr = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    folder_name = dateTimeStr+"_VO_"+testName
    # start time counter 
    path = "./" + folder_name
    os.mkdir(path)
    
    #-----------------------------------------------------------------------------------
    # [INIT VO MODULE ]
    #-----------------------------------------------------------------------------------
    # Create VisualOdometry instance 
    vo_test = VisualOdometry( path )
    
    # Settings:
    vo_test.setEnableCropping(False)
    
    # Read camera calibration paramters from file 
    vo_test.io_read_calibration((sCameraCalibrationFilePath + "calibration.yaml"))
    
    #-----------------------------------------------------------------------------------
    # [RUN TEST]
    #-----------------------------------------------------------------------------------
    # Run visual odometry 
    if not bFlagRunLiveTest:
        vo_test.run_visual_odometry_static( sTestFilePath )
    
    if bFlagRunLiveTest:
        camID = 1
        vo_test.run_visual_odometry_live(camID)
    #-----------------------------------------------------------------------------------
    # [Post-Processing and Close-out]
    #-----------------------------------------------------------------------------------
    # Create csv with VO pose estimation
    if not bFlagRunLiveTest:
        vo_test.io_writePoseToCsv()
    
    # Plot position against GT 
    if not bFlagRunLiveTest:
        vo_test.plotPosition()
    
    # close logging 
    vo_test.log.close()
    
    
if __name__=="__main__":
    main()