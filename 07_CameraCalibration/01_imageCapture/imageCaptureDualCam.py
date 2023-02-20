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

enablePatternDetector = True
enableUseBlobDetector = True

readFlags = cv2.CALIB_CB_CLUSTERING
readFlags |= cv2.CALIB_CB_ASYMMETRIC_GRID

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
boardSize = (5,15)

# Setup folder to save images to 
dateTimeStr = datetime.datetime.now().strftime("%Y_%M_%d_%h_%m_%s")
folder_name = dateTimeStr+"_frame_capture"
# start time counter 
path = "./" + folder_name

# Define imageShow window title:
WINDOW_NAME = 'StereoBench camera image feed'

# [!] Left stereo bench camera port ID
cam01_port = 1
# [!] Rigth stereo bench camera port ID
cam02_port = 0

# Flag indicating that result folder has been initialized
isFolderInit = False

# Create blob detector to support assymetric circle detection  
# TODO: currently not used. Assess if needed.       
def createBlobDetector():
    
    params = cv2.SimpleBlobDetector_Params()
    
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
    
    return cv2.SimpleBlobDetector_create(params)


# Init image index for saved files 
img_index = 0
# Open Video capture for both cameras
capture_01 = cv2.VideoCapture(cam01_port)
capture_02 = cv2.VideoCapture(cam02_port)

if enableUseBlobDetector:
    blobDetector = createBlobDetector()
else:
    blobDetector=None
                    
# Check if both webcams have been opened correctly
if not capture_01.isOpened():
    raise IOError("Cannot open webcam 01")
if not capture_02.isOpened():
    raise IOError("Cannot open webcam 02")

# Create window 
# cv2.namedWindow(WINDOW_NAME)
# cv2.startWindowThread()  
    
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
                
            # Save as concatenated image pair
            frame = cv2.hconcat([frame1, frame2])
            
            if enablePatternDetector:
                frame1_to_show = frame1
                frame2_to_show = frame2
                
                patternFound1, corners1 = cv2.findCirclesGrid( 
                                                frame1_to_show, 
                                                boardSize, 
                                                blobDetector=blobDetector,  
                                                flags=readFlags)
                patternFound2, corners2 = cv2.findCirclesGrid( 
                                                frame2_to_show, 
                                                boardSize, 
                                                blobDetector=blobDetector,  
                                                flags=readFlags)
                if patternFound1: 
                    cv2.drawChessboardCorners(frame1_to_show,
                                            boardSize,
                                            corners1,
                                            patternFound1)
                if patternFound2: 
                    cv2.drawChessboardCorners(frame2_to_show,
                                            boardSize,
                                            corners2,
                                            patternFound2)
            
                frameToDisplay = cv2.hconcat([frame1_to_show, frame2_to_show])
            else:
                frameToDisplay = frame
                
            
            # Define image name 
            img_name = "./"+folder_name+"/"+"img_"+str(img_index)+".png"
            
            # Convert stereo image pair to grayscale
            img_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            
            # Resize image for display
            display_img = cv2.resize(frameToDisplay, (1200,600), interpolation=cv2.INTER_AREA)
            
            # Show image pair 
            cv2.imshow(WINDOW_NAME, display_img)
            
            if cv2.getWindowProperty(WINDOW_NAME,cv2.WND_PROP_VISIBLE) < 1:  
                # If window has been closed -> exit    
                DoNothing = True
        # break  
        else:
            print("Image retrieval failed.")
            # Wait for key inputs
            
    # Wait for key inputs
    key = cv2.waitKey(1)
    if key == ord('q'):
        # [q] -> exit
        print('Exiting.')
        break
    elif key == 27:
        # [ESC] -> exit
        print('Exiting.')
        break
    elif key == ord('c'):
        # [c] -> Save image to file 
        if not isFolderInit :
            # Create time tagged folder to save image pairs to 
            os.mkdir(path)
            isFolderInit = True
        print()
        print("Save image to file: "+img_name)
        print()
        cv2.imwrite(img_name, img_gray) 
        img_index = img_index + 1


capture_01.release()
capture_02.release()
cv2.destroyAllWindows()
