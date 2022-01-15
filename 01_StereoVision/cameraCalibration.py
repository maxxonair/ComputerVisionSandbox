
import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image

#============================================
#       Camera calibration
#============================================

#Define size of chessboard target.
chessboard_size = (7,5)

#define criteria for subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#read calibration path folder 
calibration_paths = glob.glob('./calibration_images/*')
#Iterate over images to find intrinsic matrix

#Define arrays to save detected points
obj_points = [] #3D points in real world space 
img_points = [] #3D points in image plane

#Prepare grid and points to display
objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)

objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

isSuccess = False

for image_path in tqdm(calibration_paths):
    #Load image
    image = cv2.imread(image_path)
    #gray_image = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print('Load: ' + image_path)
    print("Image loaded, Analizying...")
    
    #find chessboard corners
    ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

    
    if ret == True:
        print("Chessboard detected!")
        print(image_path)

        #refine corner location (to subpixel accuracy) based on criteria.
        cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        isSuccess = True
    else: 
        print('No chessboard found.')

if isSuccess:
    #Calibrate camera
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_image.shape[ : : -1 ], None, None)
    #Save parameters into numpy file
    np.save("./param_camera/ret", ret)
    np.save("./param_camera/K", K)
    np.save("./param_camera/dist", dist)
    np.save("./param_camera/rvecs", rvecs)
    np.save("./param_camera/tvecs", tvecs)
else:
    print("")
    print(' >>> Calibration failed (No chessboard found)')
    print("")