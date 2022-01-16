# -------------------------------------------------------------------------------------------
#               >> Stereo Vision - 3D recontruction
# -------------------------------------------------------------------------------------------
# 
# From:
# https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
# 
# 
# 
# -------------------------------------------------------------------------------------------
import numpy as np
import glob
from tqdm import tqdm
import cv2 as cv
import matplotlib as plt
from supportFunctions import downsampleImage
# -------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------
# #Load camera parameters
ret     = np.load('./03_camera_parameters/ret.npy')
Kmat    = np.load('./03_camera_parameters/K.npy')
dist    = np.load('./03_camera_parameters/dist.npy')
#Specify input image file paths
image_LEFT_filePath = './reconstruct_this/left2.jpg'
image_RIGHT_filePath= './reconstruct_this/right2.jpg'

#Load both images
leftImage   = cv.imread(image_LEFT_filePath)
rightImage  = cv.imread(image_RIGHT_filePath)

# Left and right images MUST have identical dimensions!
h,w = leftImage.shape[:2]

#Get optimal camera matrix for better undistortion 
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(Kmat,dist,(w,h),1,(w,h))

# Undistort images
leftImage_undistorted  = cv.undistort(leftImage, Kmat, dist, None, new_camera_matrix)
rightImage_undistorted = cv.undistort(rightImage, Kmat, dist, None, new_camera_matrix)

# Reduce image size 
leftImage_downsampled  = downsampleImage(leftImage_undistorted,3)
rightImage_downsampled = downsampleImage(rightImage_undistorted,3)

#Set disparity parameters
win_size = 5
min_disp = -1
max_disp = 63 
num_disp = max_disp - min_disp 

# Needs to be divisible by 16#Create Block matching object. 
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
    numDisparities = num_disp,
    blockSize           = 5,
    uniquenessRatio     = 5,
    speckleWindowSize   = 5,
    speckleRange        = 5,
    disp12MaxDiff       = 1,
    P1 =  8*3*win_size**2,
    P2 = 32*3*win_size**2) 

print ("Generate disparity map")
disparity_map = stereo.compute(leftImage_downsampled, rightImage_downsampled)

#Show disparity map . 
plt.imshow(disparity_map,'gray')
plt.show()