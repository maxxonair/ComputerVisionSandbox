#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 20:52:04 2022

@author: x
"""

import os
import glob 
import cv2
import re
import numpy as np 
import csv
import math
import matplotlib.pyplot as plt

from util.VisualOdometryData import VisualOdometryData
from util.PyLog import PyLog
from tqdm import tqdm
from util.FeatureMatches import FeatureMatches
from util.PoseData import PoseData


class VisualOdometry:
    
    # +++++++++++++++++++++++++++++++++++++++++
    # Feature detection and matching settings
    # +++++++++++++++++++++++++++++++++++++++++
    # Feature matching Lowes ratio threshold 
    # [!] Do not touch
    # ratio_thresh = 0.2
    # minHessian   = 700
    
    ratio_thresh = 0.25
    minHessian   = 1000
    
    vo_frame_data_list = []
    
    # List holding images
    img_list = []
    
    # Mono camera intrinsic parameters
    aKmat = []
    aDist = []
    
    # +++++++++++++++++++++++++++++++++
    # Control flags
    # +++++++++++++++++++++++++++++++++
    bFlagCropRectifImages = True
    
    # +++++++++++++++++++++++++++++++++
    # Status flags
    # +++++++++++++++++++++++++++++++++
    bFlagCalibrationLoaded = False
    
    # Enable cropping after image rectification
    def setEnableCropping(self, bFlagCropRectifImages):
        self.bFlagCropRectifImages = bFlagCropRectifImages
    
    def __init__(self, sResultRootPath ):
        
        self.img_list = []
        
        # Initialize logging instance 
        flagIsConsolePrint = True
        flagIsSaveLogFile  = True
        self.log = PyLog( sResultRootPath, "VisualOdometry_log", flagIsConsolePrint, flagIsSaveLogFile)

        self.sResultRootPath            = sResultRootPath
        self.markedImagesFilePath       = sResultRootPath + "/01_marked_image_pairs/"
        self.markedSingleImageFilePath  = sResultRootPath + "/02_marked_image_singles"
        self.poseResultsFilePath        = sResultRootPath + "/03_VO_pose/"
        
        os.mkdir(self.markedImagesFilePath)
        os.mkdir(self.markedSingleImageFilePath)
        os.mkdir(self.poseResultsFilePath)
     
    #=============================================================================================
    # [INPUT/OUTPUT FUNCTIONS]
    #=============================================================================================
    
    def io_read_calibration(self, CalibrationFilePath):
        fileStorage = cv2.FileStorage()
        fileStorage.open(CalibrationFilePath, cv2.FileStorage_READ)
        
        self.log.pLogMsg('Load calibration from file.')
        
        self.aKmat = fileStorage.getNode('K_MAT').mat()
        self.aDist = fileStorage.getNode('D_VEC').mat()

        # Compute projtection matrix from camera matrix for monocular system
        self.extrinsic = np.array(((1,0,0,0),(0,1,0,0),(0,0,1,0)))
        self.aPmat = self.aKmat @ self.extrinsic
        
        fileStorage.release()
        self.bFlagCalibrationLoaded = True
            
    def io_read_images(self, sTestFilePath, sInputImgExt):
        # Empty image list 
        self.img_list = []
        
        # Create list of all files with input image file extension @ input folder 
        img_path_list = glob.glob(sTestFilePath+"/*"+sInputImgExt)

        # Sort input images by name
        img_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        self.nr_images = len( img_path_list )
        
        for ii, img_path in enumerate(img_path_list):
            self.log.pLogMsg("Read img #"+str(ii))
            self.img_list.append(cv2.imread(img_path))
            
    def io_writePoseToCsv(self):
        
        delimiter = ";"
        # open the file in the write mode
        file = open(self.poseResultsFilePath+'/pose.csv', 'w', encoding='UTF8')

        # create the csv writer
        writer = csv.writer(file)
        
        for ii, pos in enumerate( self.position_list ): 
            
            row = np.concatenate( ( np.array(pos), np.array(self.EulerAngleList[ii]) ) )
            # write a row to the csv file
            writer.writerow(row)

        # close the file
        file.close()
        
    #=============================================================================================
    # [ RUN FUNCTIONS]
    #=============================================================================================
      
    # Function: Run visual odometry on a static set of images 
    def run_visual_odometry_static(self, sTestFilePath):
        
        # Read test data 
        sInputImgExt  = ".png"
        self.log.pLogMsg("")
        self.log.pLogMsg("  [ READ test images ]  ")
        self.log.pLogMsg("")
        self.io_read_images(sTestFilePath, sInputImgExt)
        
        # Rectify input images 
        self._rectifyImgeSet()
        
        self.log.pLogMsg("")
        self.log.pLogMsg("  [ Start Viusal Odometry ]  ")
        self.log.pLogMsg("")
        
        self.PoseTransformList = []
        self.EulerAngleList    = []
        
        # Init position
        position_zero = [0, 0, 0]
        self.position_list = []
        self.position_list.append(position_zero)
        
        # List holding the incremental attitude changes between frames
        attitude_zero = np.eye(3, dtype=np.float64)
        self.dcm_list        = []
        # List holding the attitude progression wrt the starting point
        self.attitude_list   = []
        
        self.dcm_list.append(attitude_zero)
        self.attitude_list.append(attitude_zero)
        self.EulerAngleList.append(position_zero)
        
        for ii, img in tqdm( enumerate ( self.img_list ) ):
            self.log.pLogMsg("Process Image "+str(ii) + " Position" + str(self.position_list[-1]))
            self.log.pLogMsg("Attitude: " + str(self.EulerAngleList[-1]))
            # start from the second image 
            if ii > 0 :
                
                # (1) Find feature matches between frames 
                vo_frame_data, q1, q2 = self._findKeypointMatches(self.img_list[ii-1], self.img_list[ii], ii)
                
                # (2) Calculate pose change between frames from features 
                try:
                    R, t, E = self._computePose(q1, q2)   
                except:
                    self.log.pLogMsg("Computing pose failed.")
                    
                # TODO: [!] see what to do with this
                scaleFactor = 0.1
                
                # Scale translation vector to meters
                t = t * scaleFactor
                
                poseTrans = PoseData()
                poseTrans.R = R
                poseTrans.t = t
                
                self._computePoseUpdate( R, t)
                
                self.PoseTransformList.append(poseTrans)
                
                # append visual odometry data to master list 
                self.vo_frame_data_list.append(vo_frame_data)
             
        # save marked images 
        for ii, data in enumerate( self.vo_frame_data_list ):
            cv2.imwrite((self.markedImagesFilePath + "feature_match_"+str(ii)+".png"), data.img_marked_features)
    
    # Function: Connect to webcam and run visual odometry on the live video feed
    def run_visual_odometry_live(self, camID):
        
        last_img = []
        
        # Init position
        position_zero = [0, 0, 0]
        self.position_list = []
        self.position_list.append(position_zero)
        
        # List holding the incremental attitude changes between frames
        attitude_zero = np.eye(3, dtype=np.float64)
        self.dcm_list       = []
        # List holding the attitude progression wrt the starting point
        self.attitude_list   = []
        
        self.dcm_list.append(attitude_zero)
        self.attitude_list.append(attitude_zero)
        
        cap = cv2.VideoCapture(camID)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            imgCaptSuccess, img = cap.read()
            img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
            
            
            if not imgCaptSuccess:
                print("Failed to read frame")    
            else:                        
                font = cv2.FONT_HERSHEY_SIMPLEX
                yaw, pitch, roll = self._dcmTo321EulerAngles( self.attitude_list[-1] )
                # sAttitude = "yaw= "+str(yaw)+" pitch= "+str(pitch)+" roll="+str(roll)
                sAttitude = "yaw={:.2f} pitch={:.2f} roll={:.2f}".format(yaw, pitch, roll)
                cv2.putText(img, str(self.position_list[-1]), (20, 50 ), font, 1, (0,255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, sAttitude, (20, 85 ), font, 1, (0,255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Carera 01 Feed', img)      
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                # ==> [ QUIT ]
                break
            if key == ord('r'):
                # ==> [ Reset VO model ] 
                last_img = []
                # Init position
                position_zero = [0, 0, 0]
                self.position_list = []
                self.position_list.append(position_zero)
                # List holding the incremental attitude changes between frames
                attitude_zero = np.eye(3, dtype=np.float64)
                self.dcm_list       = []
                # List holding the attitude progression wrt the starting point
                self.attitude_list   = []
                self.dcm_list.append(attitude_zero)
                self.attitude_list.append(attitude_zero)
            if key == ord('u'):
                # ==> [ Update VO model ]
                ret, img = cap.read()
                img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
                
                if not ret:
                    print("Failed to read frame")        
                else:
            
                    r_img = self._rectifyImage(img)
                    
                    if len(last_img) == 0 :
                        last_img = r_img
                    else:
                        # (1) Find feature matches between frames 
                        vo_update_success = True
                        
                        try:
                            q1, q2 = self._findKeypointMatchesLight(last_img, r_img)
                        except:
                            self.log.pLogMsg("Finding matches failed.")
                            vo_update_success = False
                            
                        # (2) Calculate pose change between frames from features 
                        try:
                            R, t, E = self._computePose(q1, q2)   
                        except:
                            self.log.pLogMsg("Computing pose failed.")
                            vo_update_success = False
                        
                        if vo_update_success == True:
                            # TODO: [!] see what to do with this
                            scaleFactor = 0.1
                            
                            # Scale translation vector to meters
                            t = t * scaleFactor
                            
                            poseTrans = PoseData()
                            poseTrans.R = R
                            poseTrans.t = t
                            
                            self._computePoseUpdate( R , t )
                            
                            self.log.pLogMsg(str(self.position_list[-1]))
                            
                            last_img = r_img
            
        cap.release()
        cv2.destroyAllWindows()
        
        
        
    #=============================================================================================
    # [ INTERNAL FUNCTIONS ]
    #=============================================================================================
    
    def _rectifyImgeSet(self):
        if self.bFlagCalibrationLoaded:
            self.log.pLogMsg('')
            self.log.pLogMsg('Run [IMAGE RECTIFICATION]]')
            self.log.pLogMsg('')
            iIndex = 0
            for ii, img in tqdm( enumerate( self.img_list ) ):
                # Load image
                h,  w = img.shape[:2]

                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.aKmat,
                                                                 self.aDist,
                                                                 (w, h), 1,
                                                                 (w, h))

                # undistort
                mapx, mapy = cv2.initUndistortRectifyMap(self.aKmat,
                                                        self.aDist,
                                                        None,
                                                        newcameramtx,
                                                        (w, h),
                                                        cv2.CV_32FC1)
                
                dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

                # crop the image
                x, y, w, h = roi
                if self.bFlagCropRectifImages:
                    dst = dst[y:y+h, x:x+w]
                    
                # Write undistorted/cropped image back to image list
                self.img_list[ii] = dst

        else:
            self.log.pLogErr('Rectification aborted. No calibration data loaded.')
      
    def _rectifyImage(self, img):
        h,  w = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.aKmat,
                                                            self.aDist,
                                                            (w, h), 1,
                                                            (w, h))

        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(self.aKmat,
                                                self.aDist,
                                                None,
                                                newcameramtx,
                                                (w, h),
                                                cv2.CV_32FC1)
        
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        
        return dst
              
    def _findKeypointMatches(self, img1, img2 , frame_index):
        
        vo_frame_data = VisualOdometryData()
        #-- Step 1: Detect the keypoints using SURF Detector, 
        #           compute the descriptors
        # Initialize SURF feature detector 
        detector = cv2.xfeatures2d_SURF.create(hessianThreshold=self.minHessian)
        # Find keypoints 
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
        
        vo_frame_data.keypoints1_xy = self._keypointsToTupleList(keypoints1)
        vo_frame_data.keypoints2_xy = self._keypointsToTupleList(keypoints2)
        
        #-- Step 2: Matching descriptor vectors with a FLANN based matcher
        # Since SURF is a floating-point descriptor NORM_L2 is used
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        #-- Filter matches using the Lowe's ratio test
        vo_frame_data.feature_matches  = []
        vo_frame_data.feature_DMatches = []
        
        validKeyPoints = []
        for m,n in knn_matches:
            if m.distance < self.ratio_thresh * n.distance:
                feature_match = FeatureMatches()
                
                feature_match.keypoint_img1_xy = keypoints1[m.queryIdx].pt
                feature_match.keypoint_img2_xy = keypoints2[m.trainIdx].pt
                
                feature_match.set_px1(keypoints1[m.queryIdx].pt)
                feature_match.set_px2(keypoints2[m.trainIdx].pt)
                
                vo_frame_data.feature_matches.append(feature_match)
                vo_frame_data.feature_DMatches.append(m)
                validKeyPoints.append(m)
                
        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in validKeyPoints ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in validKeyPoints ])
        
        # Draw matches in dia-frame 
        vo_frame_data.img_marked_features = self._drawMatches(vo_frame_data, keypoints1, keypoints2, img1, img2)
        # Draw matches in initial image (img1)
        self._drawMatchesSameImage(img1 , frame_index, vo_frame_data.feature_matches )
        return vo_frame_data, q1, q2
    
    def _findKeypointMatchesLight(self, img1, img2 ):
        
        #-- Step 1: Detect the keypoints using SURF Detector, 
        #           compute the descriptors
        # Initialize SURF feature detector 
        detector = cv2.xfeatures2d_SURF.create(hessianThreshold=self.minHessian)
        # Find keypoints 
        keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
        
        #-- Step 2: Matching descriptor vectors with a FLANN based matcher
        # Since SURF is a floating-point descriptor NORM_L2 is used
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        validKeyPoints = []
        for m,n in knn_matches:
            if m.distance < self.ratio_thresh * n.distance:
                validKeyPoints.append(m)
                
        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in validKeyPoints ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in validKeyPoints ])

        return  q1, q2
    
    # Function: compute pose change between frames from keypoints in frame 1 and 2 
    def _computePose(self, q1, q2):
        
        # Compute essential matrix from key features and camera matrix 
        essentialMatrix, mask = cv2.findEssentialMat(q1 , q2 , self.aKmat)
        
        # Compute translation and rotation vector/matrix from essential matrix
        R, t = self._decomposeEssentialMatrix(essentialMatrix, q1 , q2 )
        
        # Compose transformation matrix from R and t
        transformationMatrix = self._composeTransformationMatrix(R, np.squeeze(t))
        
        return R, t, transformationMatrix
    
    def _composeTransformationMatrix(self, R, t):
        
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
        
    def _decomposeEssentialMatrix(self, essentialMatrix,  q1 , q2 ):
   
        R1, R2, t = cv2.decomposeEssentialMat(essentialMatrix)
        T1 = self._computeTransform(R1,np.ndarray.flatten(t))
        T2 = self._computeTransform(R2,np.ndarray.flatten(t))
        T3 = self._computeTransform(R1,np.ndarray.flatten(-t))
        T4 = self._computeTransform(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate((self.aKmat, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
             
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            return R2, np.ndarray.flatten(t)
        
    # Function: Form transformation matrix from given rotation matrix and translation vector
    def _computeTransform(self, R , t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = t
        return T 
         
    def _keypointsToTupleList(self, keypoints):
        keypoints_xy = []       
        for keypoint in keypoints:
            keypoints_xy.append(keypoint.pt)
        return keypoints_xy
    
    def _drawMatches(self, vo_frame_data, keypoints1, keypoints2, img1, img2 ):
        #-- Draw matches
        img_matches = np.empty((max(img1.shape[0], 
                                    img2.shape[0]), 
                                    img1.shape[1]+img2.shape[1], 3), 
                                    dtype=np.uint8)
        cv2.drawMatches(  img1, 
                            keypoints1, 
                            img2, 
                            keypoints2, 
                            vo_frame_data.feature_DMatches, 
                            img_matches, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        return img_matches 
    
    def _drawMatchesSameImage(self, img, frame_index, feature_matches):
        img_out = img
        
        # TODO: move up front 
        line_thickness = 2
        line_color = (0, 0, 255)
        for match in  feature_matches :
                
            cv2.line(  img_out, 
                       match.px1_xy, 
                       match.px2_xy, 
                       line_color, 
                       thickness=line_thickness)
            
        cv2.imwrite(self.markedSingleImageFilePath + '/frame_'+str(frame_index)+'.png', img_out)

    def _computePoseUpdate(self, R, t):
        # R  rotation from f1 -> f2
        # RT rotation from f2 -> f1
        # Get attitude dcm (reference frame to camera body frame) from previous timestep
        RR = self.attitude_list[-1]
        # Compute attitude dcm for current timestep 
        new_attitude_dcm = RR @ R 
        # Computue inverse rotation for attitude dcm at time img1
        RRT = R
        RRT.transpose()
        # translation vector in reference coordinate frame at time of img1
        t_rf = RRT @ t
        # Compute new position in the reference frame 
        position = self.position_list[-1] + t
        
        
        # Update attitude history 
        self.attitude_list.append(new_attitude_dcm)
        # Update position history 
        self.position_list.append(position)
        # Update list with incremental attitude dcm's
        self.dcm_list.append(R) 
        # Update euler angle history 
        yaw, pitch, roll = self._dcmTo321EulerAngles(new_attitude_dcm)
        self.EulerAngleList.append([yaw, pitch, roll])
        
    def _dcmToQuat(self, dcm):
        TBD = True
        
    def _dcmTo321EulerAngles(self, DCM):
        C12 = DCM[0][1]
        C11 = DCM[0][0]
        C13 = DCM[0][2]
        C23 = DCM[1][2]
        C33 = DCM[2][2]
        
        yaw    =   math.degrees( math.atan(C12/C11) )
        pitch  =   math.degrees( - math.asin(C13)   )
        roll   =   math.degrees( math.atan(C23/C33) )
        
        return yaw, pitch, roll
        
        
    def plotPosition(self):
        # TODO: move GT to read function
        gt_z = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
        gt_x = np.zeros((26,1))
        
        vo_x = np.zeros((26,1))
        vo_y = np.zeros((26,1))
        vo_z = np.zeros((26,1))
        
        vo_error_x = np.zeros((26,1))
        vo_error_y = np.zeros((26,1))
        vo_error_z = np.zeros((26,1))
        
        x_axis = np.zeros((26,1))
        
        euler_yaw   = np.zeros((26,1))
        euler_pitch = np.zeros((26,1))
        euler_roll  = np.zeros((26,1))
        
        for ii, p in enumerate( self.position_list ):
            euler = self.EulerAngleList[ii]
            vo_x[ii]        = p[0]
            vo_y[ii]        = p[1]
            vo_z[ii]        = p[2]
            vo_error_x[ii]  = abs( p[0] )
            vo_error_y[ii]  = abs( p[1] )
            vo_error_z[ii]  = abs( p[2] - gt_z[ii] )
            x_axis[ii]      = ii
            euler_yaw[ii]   = euler[0]
            euler_pitch[ii] = euler[1]
            euler_roll[ii]  = euler[2]
            
        fig = plt.figure()
        ax  = plt.gca()
        # fig.patch.set_facecolor('xkcd:mint gray')
        ax.set_facecolor((0.6, 0.6, 0.6))
        plt.subplot(3, 3, 1)
        plt.plot( x_axis, gt_x, color='blue', label='ground truth', linestyle=':', linewidth=1)
        plt.plot( x_axis, vo_x, color='red', label='VO', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('position x [m]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 4)
        plt.plot( x_axis, gt_x, color='blue', label='ground truth', linestyle=':', linewidth=1)
        plt.plot( x_axis, vo_y, color='red', label='VO', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('position y [m]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 7)
        plt.plot( x_axis, gt_z, color='blue', label='ground truth', linestyle=':', linewidth=1)
        plt.plot( x_axis, vo_z, color='red', label='VO', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('position z [m]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 2)
        plt.plot( x_axis, vo_error_x, color='red', label='VO error ', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('position error x [m]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 5)
        plt.plot( x_axis, vo_error_y, color='red', label='VO error ', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('position error y [m]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 8)
        plt.plot( x_axis, vo_error_z, color='red', label='VO error ', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('position error z [m]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 3)
        plt.plot( x_axis, euler_yaw, color='blue', label='VO YAW ', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('yaw [deg]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 6)
        plt.plot( x_axis, euler_pitch, color='blue', label='VO PITCH ', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('pitch [deg]')
        plt.legend(loc='upper left')
        
        plt.subplot(3, 3, 9)
        plt.plot( x_axis, euler_roll, color='blue', label='VO ROLL ', linestyle='-', linewidth=1)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.xlabel('it [#]')
        plt.ylabel('roll [deg]')
        plt.legend(loc='upper left')
        
        plt.show()
    