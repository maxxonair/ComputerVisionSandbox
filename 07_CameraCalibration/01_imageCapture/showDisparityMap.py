#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 17:32:21 2022

@author: x

@brief: Function to grab, display and save images from 
        two usb webcams that form a stereo camera bench.
        Image pairs will be saved as single, concatenated
        image.
        
@usage: run >./imageCaptureDualCam.py
        
"""
import cv2
from time import perf_counter
import datetime
import os
import numpy as np

# Filepath to calibration parameters
sFilePathCalibrationParameters = "../2022_09_25__08_44_41_camera_01_calibration/03_camera_parameters/stereo_calibration.yaml"

# Setup folder to save images to 
dateTimeStr = datetime.datetime.now().strftime("%Y_%M_%d_%h_%m_%s")
folder_name = dateTimeStr+"_frame_capture"
# start time counter 
path = "./" + folder_name
# Create time tagged folder to save image pairs to 
os.mkdir(path)

# Enable flipping right camera image 
# This is in case the right camera is mounted upside down
isFlipCam02image = True

# Define imageShow window title:
imgWindowName = "StereoBench camera image feed [.mk0]"

# [!] Left stereo bench camera port ID
cam01_port = 1
# [!] Rigth stereo bench camera port ID
cam02_port = 2

# Camera calibration parameters
K1 = []
D1 = []
K2 = []
D2 = []

def loadCalibrationParameters():
    global K1, D1, K2, D2
    
    fileStorage = cv2.FileStorage()
    fileStorage.open(sFilePathCalibrationParameters, cv2.FileStorage_READ)
    
    print('Load calibration from file.')
    
    K1 = fileStorage.getNode('K1').mat()
    D1 = fileStorage.getNode('D1').mat()
    K2 = fileStorage.getNode('K2').mat()
    D2 = fileStorage.getNode('D2').mat()
    
def rectifyImage(img, K, D ):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    (h, w) = img.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,
                                                        D,
                                                        (w, h), 1,
                                                        (w, h))
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(K,
                                            D,
                                            None,
                                            newcameramtx,
                                            (w, h),
                                            cv2.CV_32FC1)
    
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

def computeDepthMap(imgL, imgR):
        # SGBM Parameters -----------------
        # wsize default 3; 5; 7 for SGBM reduced size image; 
        # 15 for SGBM full size image (1300px and above); 
        window_size = 5

        left_matcher = cv2.StereoSGBM_create(
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
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.3
        visual_multiplier = 6

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)

        wls_filter.setSigmaColor(sigma)
        displ = left_matcher.compute(imgL, imgR)  
        dispr = right_matcher.compute(imgR, imgL)  
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  

        filteredImg = cv2.normalize(src=filteredImg, 
                                   dst=filteredImg, 
                                   beta=0, alpha=255, 
                                   norm_type=cv2.NORM_MINMAX)
        
        filteredImg = np.uint8(filteredImg)

        return filteredImg

def main():
    # Load calibration parameters from file 
    loadCalibrationParameters()
    # Init image index for saved files 
    img_index = 0
    # Open Video capture for both cameras
    capture_01 = cv2.VideoCapture(cam01_port)
    capture_02 = cv2.VideoCapture(cam02_port)

    # Check if both webcams have been opened correctly
    if not capture_01.isOpened():
        raise IOError("Cannot open webcam 01")
    if not capture_02.isOpened():
        raise IOError("Cannot open webcam 02")


    while True:
        # Grab frames from both cameras
        t_start = perf_counter()
        ret1 = capture_01.grab()
        t_end = perf_counter()
        ret2 = capture_02.grab()
        timetag_ms = ( t_end - t_start ) *1000
        # Print time taken to grab one cameras image
        print("Grab diff [ms] : "+str(timetag_ms))
        
        if not ret1 and not ret2:
            print("Failed to grab frames")  
            if not ret1 :
                print("Failed to grab camera 1 frame")
            if not ret2 :
                print("Failed to grab camera 2 frame")
            # TODO: Add action here       
        else:
            # Read camera frames
            suc1, frame1 = capture_01.retrieve()
            suc2, frame2 = capture_02.retrieve()
            
            if suc1 and suc2:
                
                # If selected flip right camera image
                if isFlipCam02image:
                    # Flip horizontally 
                    frame2 = cv2.flip(frame2, 0)
                    # Flip vertically ( to remove mirrored effect )
                    frame2 = cv2.flip(frame2, 1)
                    
                # Rectify camera images 
                frame1 = rectifyImage(frame1, K1, D1)
                frame2 = rectifyImage(frame2, K2, D2)
                
                # stereo    = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                disparity = computeDepthMap(frame1,frame2)
                
                # Define image name 
                img_name = "./"+folder_name+"/"+"img_"+str(img_index)+".png"
                
                # Resize image for display
                # display_img = cv2.resize(frameToDisplay, (1200,600), interpolation=cv2.INTER_AREA)
                
                # Show image pair 
                cv2.imshow(imgWindowName, disparity)
                
            else:
                print("Image retrieval failed.")
            
            # Wait for key inputs
            key = cv2.waitKey(1)
            if cv2.getWindowProperty(imgWindowName,cv2.WND_PROP_VISIBLE) < 1:  
                # If window has been closed -> exit    
                break  
            elif key == ord('q'):
                # [q] -> exit
                break
            elif key == 27:
                # [ESC] -> exit
                break
            elif key == ord('c'):
                # [c] -> Save image to file 
                print("Save image to file: "+img_name)
                cv2.imwrite(img_name, img_gray) 
                img_index = img_index + 1

    capture_01.release()
    capture_02.release()
    cv2.destroyAllWindows()
    
    
if __name__=="__main__":
    main()