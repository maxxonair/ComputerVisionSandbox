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

def main():
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
                    
                # Save as concatenated image pair
                frame = cv2.hconcat([frame1, frame2])
                
                # Define image name 
                img_name = "./"+folder_name+"/"+"img_"+str(img_index)+".png"
                
                # Convert stereo image pair to grayscale
                img_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
                
                # Resize image for display
                display_img = cv2.resize(img_gray, (1200,600), interpolation=cv2.INTER_AREA)
                
                # Show image pair 
                cv2.imshow(imgWindowName, display_img)
                
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