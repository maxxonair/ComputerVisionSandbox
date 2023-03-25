import cv2 as cv
import os 
import numpy as np


def loadStereoImage(sStereoImgPath):
    img = cv.imread(sStereoImgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    (h,w) = img.shape
    w2 = int(w/2)
    imgl = img[ 0:h , 0:(w2-1) ]
    imgr = img[ 0:h , w2: ]

    return (imgl, imgr)


def loadCalibrationParameters(sCalibrationParameterDirectory_in, log):


    fileStorage = cv.FileStorage()
    suc = fileStorage.open(sCalibrationParameterDirectory_in, cv.FileStorage_READ)
    K1=[]
    K2=[]
    D1=[]
    D2=[]
    P1=[]
    P2=[]
    R1=[]
    R2=[]
    T=[]
    Q=[]

    if suc: 
        log.pLogMsg('[x] Stereo camera calibration data set loaded.')
        K1 = fileStorage.getNode('K1').mat()
        D1 = fileStorage.getNode('D1').mat()
        K2 = fileStorage.getNode('K2').mat()
        D2 = fileStorage.getNode('D2').mat()
        P1 = fileStorage.getNode('P1').mat()
        P2 = fileStorage.getNode('P2').mat()
        R1 = fileStorage.getNode('R1').mat()
        R2 = fileStorage.getNode('R2').mat()
        T  = fileStorage.getNode('T').mat()
        Q  = fileStorage.getNode('Q').mat()

    else :
        print("Loading calibration files failed. Check file path.")
    
    # Create dictionary with all camera matrices and translation vectors 
    cameraCalibrationData = {
        'K1': K1,
        'D1': D1,
        'K2': K2,
        'D2': D2,
        'P1': P1,
        'P2': P2,
        'R1': R1,
        'R2': R2,
        'T': T,
        'Q': Q,
    }
    fileStorage.release()
    return cameraCalibrationData

def saveArrayAsCsv(array_in, sFilePath_in):
    np.savetxt(sFilePath_in, array_in, delimiter=",")

def loadStereoUndistortionMaps(sLeftUndistortionMapFilePath, sRightUndistortionMapFilePath, log):
    stacked = cv.imread(sLeftUndistortionMapFilePath, cv.IMREAD_UNCHANGED)
    
    # Check file has been found and loaded before moving on 
    if stacked is None:
        log.pLogErr('[ERR] Left camera undistortion map not found! Exiting')
        exit(1)  
    else:
        log.pLogMsg('[x] Left camera undistortion map has been found and loaded.')
        
    mapx = stacked[:,:,0:2].astype(np.int16)
    mapy = stacked[:,:,2]
    (caml_mapx, 
    caml_mapy) = cv.convertMaps(map1=mapx, map2=mapy, dstmap1type=cv.CV_32FC1)
    
    stackedr = cv.imread(sRightUndistortionMapFilePath, cv.IMREAD_UNCHANGED)
    
    # Check file has been found and loaded before moving on 
    if stackedr is None:
        log.pLogErr('[ERR] Left camera undistortion map not found! Exiting')
        exit(1) 
    else:
        log.pLogMsg('[x] Right camera undistortion map has been found and loaded.')
        
    mapx = stackedr[:,:,0:2].astype(np.int16)
    mapy = stackedr[:,:,2]
    (camr_mapx, camr_mapy) = cv.convertMaps(map1=mapx, map2=mapy, dstmap1type=cv.CV_32FC1)

    cameraCalibrationData = {
        'leftUndistortionMap_x': caml_mapx,
        'leftUndistortionMap_y': caml_mapy,
        'rightUndistortionMap_x': camr_mapx,
        'rightUndistortionMap_y': camr_mapy,
    }
    return cameraCalibrationData