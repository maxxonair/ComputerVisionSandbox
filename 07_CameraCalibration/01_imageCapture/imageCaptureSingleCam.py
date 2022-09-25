#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 17:32:21 2022

@author: x
"""
import cv2
from time import perf_counter
import datetime
import os

# Setup folder to save images to 
dateTimeStr = datetime.datetime.now().strftime("%Y_%M_%d_%h_%m_%s")
folder_name = dateTimeStr+"_frame_capture"
# start time counter 
t_start = perf_counter()
path = "./" + folder_name
os.mkdir(path)

img_index = 0 

cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to read frame")        
    else:
        # Define image name 
        t_end = perf_counter()
        timetag_ms = ( t_end - t_start ) *1000
        # Define image name with index and capture timestamp
        # img_name = "./"+folder_name+"/"+str(img_index)+"img_"+str(timetag_ms)+".png"
        # index only ([!] works with list.sort() )
        img_name = "./"+folder_name+"/"+"img_"+str(img_index)+".png"
        
        img_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Carera 01 Feed', img_gray)
    
        key = cv2.waitKey(1)
        if key == ord('q'):
            # quit 
            break
        elif key == ord('c'):
            # Save image to file 
            print("Save image to file: "+img_name)
            cv2.imwrite(img_name, img_gray) 
            img_index = img_index + 1

cap.release()
cv2.destroyAllWindows()